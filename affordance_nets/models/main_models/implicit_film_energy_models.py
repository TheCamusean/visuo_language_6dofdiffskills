import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from affordance_nets.models.common.module_attr_mixin import ModuleAttrMixin
from affordance_nets.models.common.position_encoding import PositionalEncoding2D


class ImplicitImageEBM(ModuleAttrMixin):
    def __init__(self, vision_backbone, input_dim=2, output_dim=1, p_drop_emb=0.1):
        super().__init__()

        ## Basics ##
        self.drop = nn.Dropout(p_drop_emb)

        ## Vision Model ##
        self.vision_model = vision_backbone
        self.layer = -3
        dims = self.vision_model.img_features[self.layer]

        ## EBMTransformer ##
        self.ebm_model = TransformerDecoderEBM(num_dims=dims)
        ## 2D Positional Embedding ##
        self.position_embeddings = PositionalEncoding2D(channels=dims)
        hidden_dim = 300
        self.position_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, dims))

        ## decoder head ##
        self.ln_f = nn.LayerNorm(dims)
        self.head = nn.Linear(dims, output_dim)


    def get_backbone_features(self, context):
        #with torch.no_grad():
        ## Get Visual Features from Vision Encoder ##
        images = context['images']
        visual_features = self.vision_model(images)
        return visual_features

    def train(self, x, context):
        self.set_context(context)

        n_ext = 10
        x_ext = x[:,None,:].repeat(1,n_ext,1)
        self.context_features = self.context_features[:,None,...].repeat(1, n_ext,1,1).reshape(-1,self.context_features.shape[-2], self.context_features.shape[-1])
        x_ext = x_ext.reshape(-1, x_ext.shape[-1])


        ## diffusion loss ##
        eps = torch.randn_like(x_ext)
        x_eps = x_ext + eps
        x_eps = torch.clip(x_eps, 0, 1)

        out = self.forward(x_eps).squeeze()

        dist = eps.pow(2).sum(-1).pow(.5)
        energy = torch.exp(-dist.pow(2)/0.01)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(out, energy)

        out = {'loss': loss,
               'energy': out}
        return out


    def set_context(self, context):
        ## 1. Compute Visual Features ##
        _context_features = self.get_backbone_features(context)['hidden_states'][self.layer].permute(0,2,3,1)
        pos_embedding = self.position_embeddings(_context_features)
        _context_features2 = _context_features + pos_embedding
        B = _context_features2.shape[0]
        F = _context_features2.shape[-1]
        self.context_features = _context_features2.reshape(B,-1, F)

    def forward(self, x):

        ## 2. Prepare input as [x, Features] ##
        visual_features = self.context_features


        # Set positional features as the features of the CLS token
        c_pos_emb = self.position_embeddings.get_for_query_points(x)
        position_features = self.position_embed(x)[:,None,:] + c_pos_emb

        # Run Transformer
        visual_features = self.drop(visual_features)
        position_features = self.drop(position_features)
        _out = self.ebm_model(position_features, visual_features)

        ## Layer Norm and MLP
        _out = self.ln_f(_out[:,0,:])
        out = self.head(_out)
        return out


class TransformerEncoderEBM(ModuleAttrMixin):
    def __init__(self, num_patches, num_dims,
                 n_head = 16,
                 num_layers = 3,
                 p_drop_attn = 0.1):
        super().__init__()

        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_dims,
            nhead=n_head,
            dim_feedforward=4 * num_dims,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True  # important for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, mask=None):
        if mask is None:
            out = self.encoder(src=x)
        return out


class TransformerDecoderEBM(ModuleAttrMixin):
    def __init__(self, num_dims,
                 n_head = 16,
                 num_layers = 3,
                 p_drop_attn = 0.1):
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
        out = self.decoder(tgt=x, memory = cond)
        return out




def test():

    ## Load Vision Model ##
    from affordance_nets.models.vision_backbone.clip import CLIP_Backbone
    vision_model = CLIP_Backbone()

    ## Load EBM Model ##
    ebm_model = ImplicitImageEBM(vision_backbone=vision_model)

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
    context = {'image':img_out}
    out = ebm_model(x, context)
    print(out)

    # energy_map = out.reshape(50,50)
    # plt.imshow(energy_map.detach().numpy())
    # plt.show()


if __name__ == '__main__':
    test()