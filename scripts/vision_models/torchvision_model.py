import torchvision
import torch
import torch.nn as nn
from torchvision import transforms


device = 'cuda'
convnet = torchvision.models.resnet50(pretrained=True).to(device)


normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess = nn.Sequential(
    transforms.Resize(256),
    transforms.CenterCrop(224),
    normlayer,
).to(device)

image = torch.rand(10,3,520,520)*255

image = image.to(device)
image = preprocess(image)
out = convnet(image)

print(out.shape)


