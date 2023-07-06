import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from affordance_nets.models.common.module_attr_mixin import ModuleAttrMixin
from affordance_nets.models.common.position_encoding import PositionalEncoding2D
from affordance_nets.models.core.transformers import TransformerEncoder, TransformerDecoder
from affordance_nets.models.core.cond_temporal_unet import ConditionalTrajectoryUnet1D




class ImageTrajectoryEBM(ModuleAttrMixin):
    def __init__(self, vision_backbone, input_dim=2, output_dim=2, p_drop_emb=0.1, hidden_dim=512, H=16,
                 vision_model_type='Option2'):
        super().__init__()

        ## Dimensions ##
        self.horizon = H
        self.input_dim = input_dim

        ## Basics ##
        self.drop = nn.Dropout(p_drop_emb)

        ## Vision Model ##
        self.vision_model = vision_backbone

        ## 2D Positional Embedding ##
        self.position_embeddings = PositionalEncoding2D(channels=hidden_dim)

        ## Vision to Features ##
        self.layer = -1
        dims = self.vision_model.img_features[self.layer]
        img_scale = self.vision_model.img_layers[self.layer]
        self.vision_2_features = nn.Conv2d(dims, hidden_dim, kernel_size=1)

        self.vision_model_type = vision_model_type
        self.transformer_input = torch.nn.Parameter(torch.randn(H, hidden_dim), requires_grad=True)
        self.transformer_model = TransformerDecoder(num_dims=hidden_dim, num_layers=5)


        ## Combine Vision and Control ##
        self.model = ConditionalTrajectoryUnet1D(input_dim=input_dim,
                                                 cond_dim=hidden_dim)


    def get_backbone_features(self, context):
        with torch.no_grad():
            ## Get Visual Features from Vision Encoder ##
            images = context['images']
            visual_features = self.vision_model(images)
            return visual_features


    def set_context(self, context):
        ## 1. Compute Visual Features ##
        _context_features = self.get_backbone_features(context)['hidden_states'][self.layer]
        _context_features = self.vision_2_features(_context_features)
        pos_embedding = self.position_embeddings(_context_features)
        _context_features2 = _context_features + pos_embedding
        B = _context_features2.shape[0]
        F = _context_features2.shape[1]
        _context = _context_features2.reshape(B, F, -1)
        in_context = _context.permute(0,2,1)

        if self.vision_model_type=='Option1':
            context_features = self.transformer_model(in_context)
            _context = context_features.permute(0,2,1)
            _context = self.drop(_context)
            self.context_features = self.avg_pool(_context).squeeze(-1)
        else:
            input_ext = self.transformer_input[None,...].repeat(B,1,1)
            self.context_features = self.transformer_model(x=input_ext, cond=in_context).squeeze(1)

    def train(self, x, context):
        self.set_context(context)

        n_ext = 100
        x_ext = x[:,None,...].repeat(1, n_ext, 1, 1)
        self.context_features = self.context_features[:,None,...].repeat(1, n_ext,1,1).reshape(-1, self.context_features.shape[-2],
                                                                                               self.context_features.shape[-1])
        x_ext = x_ext.reshape(-1, x_ext.shape[-2], x_ext.shape[-1])


        ## diffusion loss ##
        eps = torch.randn_like(x_ext)
        x_eps = x_ext + eps
        #x_eps = torch.clip(x_eps, 0, 1)

        out = self.forward(x_eps).squeeze()

        dist = eps.pow(2).sum(-1).pow(.5)
        energy = torch.exp(-dist.pow(2)/0.01)

        #loss_fn = torch.nn.HuberLoss(delta=1.)
        loss_fn = torch.nn.L1Loss()
        loss = loss_fn(out, eps)

        out = {'loss': loss,
               'energy': out}
        return out


    def sample(self, B=1, T=500, alpha=1e-2, alpha_0=0.):
        x0 = torch.rand(B, self.horizon, self.input_dim).to(self.context_features)
        self.context_features = self.context_features.repeat(B, 1, 1)

        _x = x0
        for t in range(T):
            alpha_t = alpha_0*(T-t)/T + alpha
            dx = self.forward(_x)
            _x = _x - dx*alpha_t
        return _x

    def forward(self, x):

        ## Visual Features ##
        visual_features = self.context_features

        # Run FilM conditioning
        out = self.model(x, visual_features)
        return out



def test():

    ## Load Vision Model ##
    from affordance_nets.models.vision_backbone.resnet import ResNet18_Backbone
    vision_model = ResNet18_Backbone()

    ## Load EBM Model ##
    ebm_model = ImageTrajectoryEBM(vision_backbone=vision_model)

    ## Evaluation ##
    from PIL import Image
    import requests
    import time
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    x = torch.randn(1, 16, 2)
    ## Process Image ##
    img_out = ebm_model.vision_model.image_preprocess(image)
    ## EBM ##
    context = {'images':img_out}

    for k in range(100):
        t0 = time.time()
        ebm_model.set_context(context)
        t1 = time.time()
        out = ebm_model(x)
        print('Times 0: {}, Time 1: {}'.format(time.time()- t1, t1-t0))
        print(out.shape)


if __name__ == '__main__':
    test()