from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
#from datasets import load_dataset
from PIL import Image
import requests

device = 'cuda'

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18").to(device)

inputs = feature_extractor(image, return_tensors="pt").to(device)


_out = model(**inputs ,output_hidden_states=True)
print(_out['hidden_states'][-1].shape)


with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])