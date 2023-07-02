import torch
import torch.nn as nn
from affordance_nets.models.vision_backbone.vision_base import BaseVisionBackbone, BaseVisionTransformerBackbone

from transformers import CLIPProcessor, CLIPModel


class CLIP_Backbone(BaseVisionBackbone):

    def __init__(self):

        img_layers = [7]*13
        img_features = [768]*13
        img_size = 224

        super().__int__(img_layers=img_layers, img_features=img_features, img_size=img_size)

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = model.vision_model

        ## Preproccess Image ##
        self.image_adaptor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def image_preprocess(self, image):
        out = self.image_adaptor(images=image, return_tensors="pt")
        return out['pixel_values']

    def forward(self, image, preproccess=False):
        if preproccess:
            image = self.image_preprocess(image)

        B = image.shape[0]
        features = self.vision_model(pixel_values=image, output_hidden_states= True)

        hidden_states = []
        for k in range(len(features['hidden_states'])):
            hidden_states.append(features['hidden_states'][k][:,1:,:].permute(0,2,1).reshape(B, self.img_features[k], self.img_layers[k], self.img_layers[k]))
        features['hidden_states'] = hidden_states
        return features




def test():
    vision_model = CLIP_Backbone()

    from PIL import Image
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    vision_model(image, preproccess=True)





if __name__ == '__main__':
    test()