import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from affordance_nets.models.common.module_attr_mixin import ModuleAttrMixin
from affordance_nets.models.common.position_encoding import PositionalEncoding2D


class ImplicitTransformerImageEBM(ModuleAttrMixin):
    def __init__(self, vision_backbone, input_dim=2, output_dim=2, p_drop_emb=0.1, hidden_dim=512):
        super().__init__()

        ## Basics ##
        self.drop = nn.Dropout(p_drop_emb)

        ## Move X to Latent ##
        self.bottom = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish())

        ## Vision Model ##
        self.vision_model = vision_backbone

        ## 2D Positional Embedding ##
        self.position_embeddings = PositionalEncoding2D(channels=hidden_dim)

        ## Vision to Features ##
        self.layer = -1
        dims = self.vision_model.img_features[self.layer]
        img_scale = self.vision_model.img_layers[self.layer]
        self.vision_2_features = nn.Conv2d(dims, hidden_dim, kernel_size=1)

        ## Main Transformer Network ##
        self.transformer_model = TransformerDecoder(num_dims=hidden_dim)

        ## decoder head ##
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)


    def get_backbone_features(self, context):
        with torch.no_grad():
            ## Get Visual Features from Vision Encoder ##
            images = context['images']
            visual_features = self.vision_model(images)
            return visual_features

    def train(self, x, context):
        self.set_context(context)

        n_ext = 100
        x_ext = x[:,None,:].repeat(1,n_ext,1)
        self.context_features = self.context_features[:,None,...].repeat(1, n_ext,1, 1).reshape(-1,
                                                                                                self.context_features.shape[-2],
                                                                                                self.context_features.shape[-1])
        x_ext = x_ext.reshape(-1, x_ext.shape[-1])


        ## diffusion loss ##
        eps = torch.randn_like(x_ext)
        x_eps = x_ext + eps
        x_eps = torch.clip(x_eps, 0, 1)

        out = self.forward(x_eps).squeeze()

        dist = eps.pow(2).sum(-1).pow(.5)
        energy = torch.exp(-dist.pow(2)/0.01)

        #loss_fn = torch.nn.HuberLoss(delta=1.)
        loss_fn = torch.nn.L1Loss()
        loss = loss_fn(out, eps)

        out = {'loss': loss,
               'energy': out}
        return out

    def set_context(self, context):
        ## 1. Compute Visual Features ##
        _context_features = self.get_backbone_features(context)['hidden_states'][self.layer]#.permute(0,2,3,1)
        _context_features = self.vision_2_features(_context_features)
        pos_embedding = self.position_embeddings(_context_features)
        _context_features2 = _context_features + pos_embedding
        B = _context_features2.shape[0]
        F = _context_features2.shape[1]
        _context = _context_features2.reshape(B, F, -1)
        self.context_features = _context.permute(0,2,1)

    def forward(self, x):

        ## Visual Features ##
        visual_features = self.context_features

        # Set positional features as the features of the CLS token
        out = self.bottom(x)[:,None,:]
        #out = self.position_embeddings.get_for_query_points(x, scale_x=7, scale_y=7)

        # Run FilM conditioning
        visual_features = self.drop(visual_features)
        out = self.transformer_model(out, visual_features)[:,0,:]

        ## Layer Norm and MLP
        out = self.ln_f(out)
        out = self.head(out)
        return out



class TransformerDecoder(ModuleAttrMixin):
    def __init__(self, num_dims,
                 n_head = 16,
                 num_layers = 5,
                 p_drop_attn = 0.01):
        super().__init__()

        # encoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=num_dims,
            nhead=n_head,
            dim_feedforward=4 * num_dims,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, cond, mask=None):
        out = self.decoder(tgt=x, memory=cond)
        return out




def test():

    ## Load Vision Model ##
    from affordance_nets.models.vision_backbone.resnet import ResNet18_Backbone
    vision_model = ResNet18_Backbone()

    ## Load EBM Model ##
    ebm_model = ImplicitTransformerImageEBM(vision_backbone=vision_model)

    ## Evaluation ##
    from PIL import Image
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    x = torch.randn(1,2)
    ## Process Image ##
    img_out = ebm_model.vision_model.image_preprocess(image)
    ## EBM ##
    context = {'images':img_out}

    ebm_model.set_context(context)
    out = ebm_model(x)
    print(out)


if __name__ == '__main__':
    test()