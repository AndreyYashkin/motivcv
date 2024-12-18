from typing import override
import torch
import torch.nn as nn

# import torchvision.transforms.functional as F


class Normalize(nn.Module):
    def __init__(self, mean, std, inplace=True):
        super().__init__()
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.inplace = inplace

    @override
    def forward(self, image):
        # return F.normalize(image, self.mean, self.std, self.inplace)
        mean = self.mean.view(-1, 1, 1)
        std = self.std.view(-1, 1, 1)
        if self.inplace:
            return image.sub_(mean).div_(std)
        else:
            return image.sub(mean).div(std)


class ToGrayscale(nn.Module):
    def __init__(self, keep_channels):
        super().__init__()
        self.keep_channels = keep_channels

    @override
    def forward(self, image):
        image = torch.mean(image, dim=1, keepdim=True)
        if self.keep_channels:
            image = torch.repeat_interleave(image, 3, dim=1)
        return image
