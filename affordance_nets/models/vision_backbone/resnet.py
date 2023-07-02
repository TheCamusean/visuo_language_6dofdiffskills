import torch
import torch.nn as nn
from affordance_nets.models.vision_backbone.vision_base import BaseVisionBackbone

from transformers import ResNetForImageClassification, AutoFeatureExtractor


class ResNet18_Backbone(BaseVisionBackbone):

    def __init__(self):

        img_layers = [56, 56, 28, 14, 7]
        img_features = [64, 64, 128, 256, 512]
        img_size = 224

        super().__int__(img_layers=img_layers, img_features=img_features, img_size=img_size)

        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")

        self.vision_model = model

        ## Preproccess Image ##
        self.image_adaptor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
        self.image_adaptor.crop_pct = 1.

    def image_preprocess(self, image):
        out = self.image_adaptor(images=image, return_tensors="pt")
        return out['pixel_values']

    def forward(self, image, preproccess=False, output_hidden_states=True):
        if preproccess:
            image = self.image_preprocess(image)

        features = self.vision_model(pixel_values=image, output_hidden_states= output_hidden_states)
        return features




def test():
    vision_model = ResNet18_Backbone()

    from PIL import Image
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    vision_model(image, preproccess=True)





if __name__ == '__main__':
    test()