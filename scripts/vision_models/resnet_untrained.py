import torch
import torchvision
import torch.nn as nn

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


vision_encoder = get_resnet('resnet18')


device = 'cuda'
img = torch.rand(1, 3, 500, 500).to(device)
out = vision_encoder(img)
print(out.shape)