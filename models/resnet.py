"""Resnet18 model."""

import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

from kernels.accessor import get_kernel
from models.resnet_block import ResnetBlock
from utils.globals import *

class Resnet(nn.Module):
    """ResNet 18 model."""

    model_data = pd.DataFrame(columns=[
        'Data Mean', 'Data STD',
        'Layer 1 Mean', 'Layer 1 STD',
        'Layer 5 Mean', 'Layer 5 STD',
        'Layer 9 Mean', 'Layer 9 STD',
        'Layer 13 Mean', 'Layer 13 STD',
        'Layer 18 Mean', 'Layer 18 STD'
    ])

    def __init__(self, channels_in: int, channels_out: int):
        """Initialize Resnet18 model.

        Args:   
            channels_out (int): Input channels (Likely 1 if images are B&W, 3 if colored)
            channels_out (int): Output channels (number of classes in dataset)
        """
        super(Resnet, self).__init__()

        # Initialize attributes
        self.std0 = self.std1 = (ARGS.scale if ARGS.distribution != 'poisson' else ARGS.rate)
        self.mean1 = 1
        self.planes_in = 64

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)

        # Convolving layers
        self.conv1 = nn.Conv2d(channels_in, 64, kernel_size=3, stride=1, padding=1, bias=False)

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

        if ARGS.debug: LOGGER.debug(f"RESNET X1: {x1.shape}")

        x2 = self.layer1(x1)

        if ARGS.debug: LOGGER.debug(f"RESNET X2: {x2.shape}")
        x3 = self.layer2(x2)

        if ARGS.debug: LOGGER.debug(f"RESNET X3: {x3.shape}")
        x4 = self.layer3(x3)

        if ARGS.debug: LOGGER.debug(f"RESNET X4: {x4.shape}")
        x5 = F.avg_pool2d(self.layer4(x4), 4)

        if self.training: self.record_params(X, x1, x2, x3, x4, x5)

        return self.linear(x5.view(x5.size(0), -1))
    
    def update_kernels(self, epoch: int) -> None:
        """Update kernels."

        Args:
            epoch (int): Current epoch
        """
        self.kernel1 = get_kernel(64)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for child in layer:
                child.update_kernels()

    def record_params(self, x, x1, x2, x3, x4, x5):
        """Record model's location and scale parameters.

        Args:
            x (torch.Tensor): Intermediate tensor
            x1 (torch.Tensor): Intermediate tensor
            x2 (torch.Tensor): Intermediate tensor
            x3 (torch.Tensor): Intermediate tensor
            x4 (torch.Tensor): Intermediate tensor
            x5 (torch.Tensor): Intermediate tensor
        """
        with torch.no_grad():
            self.model_data.loc[len(self.model_data)] = [
                round(torch.mean(  x.float()).item(), 3),
                round(torch.std(   x.float()).item(), 3),
                round(torch.mean( x1.float()).item(), 3),
                round(torch.std(  x1.float()).item(), 3),
                round(torch.mean( x2.float()).item(), 3),
                round(torch.std(  x2.float()).item(), 3),
                round(torch.mean( x3.float()).item(), 3),
                round(torch.std(  x3.float()).item(), 3),
                round(torch.mean( x4.float()).item(), 3),
                round(torch.std(  x4.float()).item(), 3),
                round(torch.mean( x5.float()).item(), 3),
                round(torch.std(  x5.float()).item(), 3)
            ]

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

    def _make_layer(self, planes_out: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create ResnetBlock layer.

        Args:
            planes_out (int): Number of output planes
            num_blocks (int): Number of blocks for layer
            stride (int): Kernel convolution stride

        Returns:
            nn.Sequential: ResnetBlock layer
        """
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResnetBlock(self.planes_in, planes_out, stride))
            self.planes_in = planes_out * ResnetBlock.expansion
        return nn.Sequential(*layers)

    def to_csv(self, file_path: str) -> None:
        """Dump all data to CSV file.

        Args:
            file_path (str): Path at which model data will be saved
        """
        self.model_data.to_csv(file_path)