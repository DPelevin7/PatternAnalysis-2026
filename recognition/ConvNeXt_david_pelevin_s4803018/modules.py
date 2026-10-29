import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A simple convolutional neural network for 256x240px images

    Code developed using the following websites
    https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    def __init__(self):
        """
        Initialises the components of the CNN with the relevant dimensions
        """
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(5824, 128)
        
        self.fc2 = nn.Linear(128, 2)

    def forward(self, input):
        """
        Classifies a batch of images using the CNN and returns logits
        representing the confidence of the model in each class.

        Parameters:
            input (Tensor N, C, H, W / C, H, W): The batch of images to be categorised.
        """
        input = self.pool(F.relu(self.conv1(input)))
        input = self.pool(F.relu(self.conv2(input)))
        input = self.pool(F.relu(self.conv3(input)))
        input = self.pool(F.relu(self.conv4(input)))

        input = torch.flatten(input, 1)
        input = F.relu(self.fc1(input))
        input = self.fc2(input)
        return input


## ConvNeXt architecture based on https://www.geeksforgeeks.org/computer-vision/convnext/
class ConvNeXtBlock(nn.Module):
    def __init__(self, dimension, layer_scale=1e-6):
        super().__init__()

        # Depthwise convolution settings found in
        # https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html
        self.depthConv2d = nn.Conv2d(dimension, dimension, 7, padding=3, groups=dimension)

        self.l1 = nn.LayerNorm(dimension)
        self.conv1 = nn.Conv2d(dimension, dimension * 4, 1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(dimension * 4, dimension, 1)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dimension))

    def forward(self, input):
        carryThrough = input

        input = self.depthConv2d(input)
        input = self.l1(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input = self.conv1(input)
        input = self.gelu(input)
        input = self.conv2(input)
        input = (self.gamma * input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # Stochastic Depth A.K.A Drop Path
        input = torchvision.ops.stochastic_depth(input, 0.1, "row", self.training)

        return input + carryThrough


class downSample(nn.Module):
    def __init__(self, dimension):
        super().__init__()

        self.l1 = nn.LayerNorm(dimension)
        self.conv1 = nn.Conv2d(dimension, dimension * 2, 2, 2)

    def forward(self, input):
        input = self.l1(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input = self.conv1(input)
        return input


class ConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 96, 4, 4)
        self.l1 = nn.LayerNorm(96)

        self.convnext1 = nn.Sequential(
            ConvNeXtBlock(96),
            ConvNeXtBlock(96),
            ConvNeXtBlock(96)
        )

        self.downSample1 = downSample(96)

        self.convnext2 = nn.Sequential(
            ConvNeXtBlock(192),
            ConvNeXtBlock(192),
            ConvNeXtBlock(192)
        )

        self.downSample2 = downSample(192)

        self.convnext3 = nn.Sequential(
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384),
            ConvNeXtBlock(384)
        )

        self.downSample3 = downSample(384)

        self.convnext4 = nn.Sequential(
            ConvNeXtBlock(768),
            ConvNeXtBlock(768),
            ConvNeXtBlock(768)
        )

        self.averagePool = nn.AdaptiveAvgPool2d(1)

        self.l2 = nn.LayerNorm(768)

        self.linear = nn.Linear(768, 2)

    def forward(self, input):
        input = self.conv1(input)
        input = self.l1(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        input = self.convnext1(input)
        input = self.downSample1(input)
        input = self.convnext2(input)
        input = self.downSample2(input)
        input = self.convnext3(input)
        input = self.downSample3(input)
        input = self.convnext4(input)
        input = self.averagePool(input)
        input = self.l2(input.permute(0, 2, 3, 1))
        input = self.linear(input).permute(0, 3, 1, 2)
        input = torch.flatten(input, 1)
        return input