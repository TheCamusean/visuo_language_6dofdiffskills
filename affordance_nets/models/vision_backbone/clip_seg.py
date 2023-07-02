import torch
import torch.nn as nn
from affordance_nets.models.vision_backbone.vision_base import BaseVisionBackbone

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


class CLIPSeg_Backbone(BaseVisionBackbone):

    def __init__(self):

        img_layers = [22]*13
        img_features = [768]*13
        img_size = 352

        super().__int__(img_layers=img_layers, img_features=img_features, img_size=img_size)


        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.vision_model = model.clip.vision_model

        ## Preproccess Image ##
        self.image_adaptor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    def image_preprocess(self, image):
        out = self.image_adaptor(images=image, return_tensors="pt")
        return out['pixel_values']

    def forward(self, image, preproccess=False, output_hidden_states=True):
        if preproccess:
            image = self.image_preprocess(image)

        B = image.shape[0]
        features = self.vision_model(pixel_values = image, output_hidden_states= output_hidden_states)

        hidden_states = []
        for k in range(len(features['hidden_states'])):
            hidden_states.append(features['hidden_states'][k][:,1:,:].reshape(B, self.img_layers[k], self.img_layers[k], self.img_features[k]))
        features['hidden_states'] = hidden_states
        return features




def test():
    vision_model = CLIPSeg_Backbone()

    from PIL import Image
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    vision_model(image, preproccess=True)





if __name__ == '__main__':
    test()