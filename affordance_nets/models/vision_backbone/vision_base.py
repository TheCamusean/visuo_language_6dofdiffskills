import torch
import torch.nn as nn


class BaseVisionBackbone(nn.Module):

    def __int__(self, img_size, img_layers, img_features):
        super(BaseVisionBackbone, self).__init__()

        self.type = 'resnet'

        self.img_layers = img_layers
        self.img_features =  img_features
        self.img_size = img_size

    def image_preprocess(self, image):
        print('To be define')


class BaseVisionTransformerBackbone(nn.Module):

    def __int__(self, num_patches, num_dim, img_size):
        super(BaseVisionTransformerBackbone, self).__init__()

        self.type = 'transformer'

        self.num_patches = num_patches
        self.num_dim = num_dim
        self.img_size = img_size

    def image_preprocess(self, image):
        print('To be define')