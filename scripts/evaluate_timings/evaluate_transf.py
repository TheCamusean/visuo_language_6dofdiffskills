import time

import torch
import torch.nn as nn

device = 'cuda'

########### Transformer Decoder ###############
num_dims = 512
n_head = 16
p_drop_attn = 0.1
num_layers = 5

decoder_layer = nn.TransformerDecoderLayer(
    d_model=num_dims,
    nhead=n_head,
    dim_feedforward=4 * num_dims,
    dropout=p_drop_attn,
    activation='gelu',
    batch_first=True,
    norm_first=True  # important for stability
)
encoder = nn.TransformerDecoder(
    decoder_layer=decoder_layer,
    num_layers=num_layers
).to(device)



for k in range(1000):
    img = torch.randn(1, 401, num_dims).to(device)
    x = torch.randn(1, 10, num_dims).to(device)
    st_time = time.time()
    out = encoder(x, img)
    print('time taken {}'.format(time.time() - st_time))