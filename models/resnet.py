"""Resnet18 model."""

import torch, torch.nn as nn, torch.nn.functional as F

from models.resnet_block import ResnetBlock

class Resnet(nn.Module):

    def __init__(self, channels_out: int, num_blocks: int):
        """Initialize Resnet18 model.

        Args:   
            channels_out (int): Output channels (number of classes in dataset)
            num_blocks (int): Number of Resnet blocks
        """
        super(Resnet, self).__init__()

        # Initialize attributes
        self.planes_in = 64

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)

        # Convolving layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Block layers
        self.layer1 = self._make_layer( 64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Linear layers
        self.linear = nn.Linear(512*ResnetBlock.expansion, channels_out)

        self._initialize_weights()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Feed input through network and return output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x1 = self.conv1(X)

        with torch.no_grad():
         y = x1.float()
         self.std1 = torch.std(y).item()
         self.mean1 = torch.mean(y).item()

        x1 = F.relu(self.bn1(self.kernel1(x1)))

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)

        x5 = F.avg_pool2d(self.layer4(x4), 4)

        if self.modelStatus:
            self.get_std(X, x1, x2, x3, x4, x5)

        return self.linear(x5.view(x5.size(0), -1))

    def _initialize_weights(self) -> None:
        """Initialize weights in all layers.
        """
        for m in self.modules():

            # Convolving layers
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # Batch normalization layers
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # Linear layers
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create ResnetBlock layer.

        Args:
            planes (int): Number of input planes
            num_blocks (int): Number of blocks for layer
            stride (int): Kernel convolution stride

        Returns:
            nn.Sequential: ResnetBlock layer
        """
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResnetBlock(self.in_planes, planes, stride))
            self.in_planes = planes * ResnetBlock.expansion
        return nn.Sequential(*layers)