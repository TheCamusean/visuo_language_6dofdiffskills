import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from affordance_nets.models.common.module_attr_mixin import ModuleAttrMixin
from affordance_nets.models.common.position_encoding import PositionalEncoding2D


class ImplicitFiLMImageEBM(ModuleAttrMixin):
    def __init__(self, vision_backbone, input_dim=2, output_dim=2, p_drop_emb=0.1,
                      hidden_dim=248):
        super().__init__()

        ## Basics ##
        self.drop = nn.Dropout(p_drop_emb)

        ## Move X to Latent ##
        self.bottom = nn.Linear(input_dim, hidden_dim)

        ## Vision Model ##
        self.vision_model = vision_backbone

        self.layer = -1
        dims = self.vision_model.img_features[self.layer]
        img_scale = self.vision_model.img_layers[self.layer]
        img_size_out = (img_scale)

        self.vision_rescale = nn.Conv2d(dims, 700, kernel_size=3, stride=img_size_out)
        vision_dim = 700

        ## FiLM conditioned network ##
        self.film_model = FilmModel(input_dim= hidden_dim,
                                   cond_dim = vision_dim,
                                   output_dim = hidden_dim,
                                   num_layers=5)

        ## decoder head ##
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def get_backbone_features(self, context):
        #with torch.no_grad():
        ## Get Visual Features from Vision Encoder ##
        images = context['images']
        visual_features = self.vision_model(images)
        return visual_features

    def train(self, x, context):
        self.set_context(context)

        n_ext = 100
        x_ext = x[:,None,:].repeat(1,n_ext,1)
        self.context_features = self.context_features[:,None,...].repeat(1, n_ext,1).reshape(-1, self.context_features.shape[-1])
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
        _context_features = self.get_backbone_features(context)['hidden_states'][self.layer]
        _context_features = self.vision_rescale(_context_features)
        B = _context_features.shape[0]
        self.context_features = _context_features.reshape(B, -1)

    def forward(self, x):

        visual_features = self.context_features

        # Set positional features as the features of the CLS token
        out = self.bottom(x)

        # Run FilM conditioning
        visual_features = self.drop(visual_features)
        out = self.film_model(out, visual_features)

        ## Layer Norm and MLP
        out = self.ln_f(out)
        out = self.head(out)
        return out

class FilmModel(ModuleAttrMixin):
    def __init__(self,  input_dim,
                        cond_dim,
                        output_dim,
                        num_layers=3,
                        hidden_dims=None):
        super().__init__()

        self.num_layers = num_layers

        dims = [input_dim]
        if hidden_dims is None:
            dims = dims + [input_dim]*(num_layers-1)
        dims = dims + [output_dim]

        self.main_modules = nn.ModuleList([])
        for k in range(num_layers):
            self.main_modules.append(ResidualFiLM(
                input_dim=dims[k],
                output_dim=dims[k+1],
                cond_dim=cond_dim,
            ))

    def forward(self, x, cond):
        out = x
        for k in range(self.num_layers):
            out = self.main_modules[k](out, cond)
        return out


class ResidualFiLM(ModuleAttrMixin):
    def __init__(self,  input_dim,
                        cond_dim,
                        output_dim,
                        hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.blocks = nn.ModuleList([
            nn.Mish(),
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        self.hidden_dim = hidden_dim
        cond_channels = hidden_dim * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        self.head = nn.Linear(input_dim, output_dim)

    def forward(self, x, cond):

        x = self.blocks[0](x)

        out = self.blocks[1](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.hidden_dim)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.head(x)
        return out


def test():

    ## Load Vision Model ##
    from affordance_nets.models.vision_backbone.resnet import ResNet18_Backbone
    vision_model = ResNet18_Backbone()

    ## Load EBM Model ##
    ebm_model = ImplicitFiLMImageEBM(vision_backbone=vision_model)

    ## Evaluation ##
    from PIL import Image
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    vision_model(image, preproccess=True)


    x = torch.linspace(0,1,50)
    xx,yy = torch.meshgrid((x,x))
    x = torch.cat((xx.reshape(-1)[:,None], yy.reshape(-1)[:,None]), dim=-1)

    ## Process Image ##
    img_out = ebm_model.vision_model.image_preprocess(image)
    ## EBM ##
    context = {'images':img_out}

    ebm_model.set_context(context)
    out = ebm_model(x)
    print(out)

    # energy_map = out.reshape(50,50)
    # plt.imshow(energy_map.detach().numpy())
    # plt.show()


if __name__ == '__main__':
    test()