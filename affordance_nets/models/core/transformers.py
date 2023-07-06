import torch
import torch.nn as nn
from affordance_nets.models.common.module_attr_mixin import ModuleAttrMixin



class TransformerEncoder(ModuleAttrMixin):
    def __init__(self, num_dims,
                 n_head = 16,
                 num_layers = 5,
                 p_drop_attn = 0.01):
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
        out = self.encoder(x)
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