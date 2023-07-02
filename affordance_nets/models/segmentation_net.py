from typing import Any, Optional, Tuple, Union
import math

import torch
from torch import nn

from affordance_nets.models.unet_basics import Up
from affordance_nets.models.resnet_basics import ConvBlock, IdentityBlock


class SegmentationNet(nn.Module):

    def __init__(self, device='cpu', output_channels=1):
        super().__init__()

        self.visual_encoder = self.load_visual_encoder_hf(device)
        self.extract_layers = [2,3,4]

        self.decoder = VisualResNetDecoder(output_channels=output_channels).to(device)

    def load_visual_encoder_hf(self, device):
        from transformers import ResNetForImageClassification

        vision_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18").to(device)
        return vision_model

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
    ):

        # step 1: forward the query images through the frozen vision encoder
        #with torch.no_grad():
        vision_outputs = self.visual_encoder(pixel_values, output_hidden_states=True)
        hidden_states = vision_outputs.hidden_states
        # we add +1 here as the hidden states also include the initial embeddings
        activations = [hidden_states[i] for i in self.extract_layers]


        # step 2: forward the activations through the lightweight decoder to predict masks
        decoder_outputs = self.decoder(activations,)
        logits = decoder_outputs['logits']

        loss = None
        if labels is not None:
            loss = self.get_loss(logits, labels)
            # # move labels to the correct device to enable PP
            # labels = labels.to(logits)
            # loss_fn = nn.BCEWithLogitsLoss()
            # loss = loss_fn(logits, labels)

        if not return_dict:
            output = (logits, vision_outputs, decoder_outputs)
            return ((loss,) + output) if loss is not None else output

    def get_loss(self, logits, labels):
        labels = labels.to(logits)
        loss_fn = nn.MSELoss()
        loss = loss_fn(torch.sigmoid(logits), labels)
        return loss


class VisualResNetDecoder(nn.Module):
    def __init__(self, output_channels=1):
        super().__init__()

        self.input_dim = 512

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.up1 = Up(512, 256)

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.up2 = Up(256, 128)


        self.batchnorm = True
        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.output_dim = output_channels
        self.conv_out = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )


    def forward(
        self,
        hidden_states: Tuple[torch.Tensor],
    ):
        activations = hidden_states[::-1]

        ## Unet Upscaling ##
        x = self.conv1(activations[0])
        x = self.up1(x, activations[1])
        x = self.conv2(x)
        x = self.up2(x, activations[2])

        ## Upscale to image size ##
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        logits = self.conv_out(x)

        return {'logits': logits.squeeze(1)}


if __name__ == '__main__':
    model = SegmentationNet()

    from transformers import AutoFeatureExtractor, ResNetForImageClassification
    import torch
    from PIL import Image
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    inputs = feature_extractor(image, return_tensors="pt")


    out = model(**inputs)
    print(out.shape)