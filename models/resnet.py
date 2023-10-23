"""Resnet 18 model."""

import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

from kernels.accessor import get_kernel
from utils import ARGS, LOGGER
from models.resnet_block import ResnetBlock

class Resnet(nn.Module):
    """ResNet 18 model."""

    # Initialize logger
    logger = LOGGER.getChild('resnet')

    # Initialize layer data file
    model_data = pd.DataFrame(columns=[
        'Data-STD',     'Data-Mean',
        'Layer 1-STD',  'Layer 1-MEAN',
        'Layer 5-STD',  'Layer 5-MEAN',
        'Layer 9-STD',  'Layer 9-MEAN',
        'Layer 13-STD', 'Layer 13-MEAN',
        'Layer 18-STD', 'Layer 18-MEAN'
    ])

    def __init__(self, channels_in: int, channels_out: int):
        """Initialize ResNet18 model.

        Args:
            channels_in (int): Input channels
            channels_out (int): Output channels
        """
        super(Resnet, self).__init__()

        # Initialize distribution parameters
        self.planes_in =    64
        self.location =     [1]*6
        self.scale =        [1]*6
        self.rate =         [1]*6

        # Batch normalization layer
        self.bn =           nn.BatchNorm2d(64)

        # Convolving layer
        self.conv =         nn.Conv2d(channels_in, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Block layers
        self.layer1 =       self._make_layer( 64, 2, stride=1)
        self.layer2 =       self._make_layer(128, 2, stride=2)
        self.layer3 =       self._make_layer(256, 2, stride=2)
        self.layer4 =       self._make_layer(512, 2, stride=2)

        # Linear layer
        self.linear =       nn.Linear(512*ResnetBlock.expansion, channels_out)

        self._initialize_weights()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Feed input through network and return output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        if ARGS.debug: self.logger.debug(f"Input shape: {X.shape}")

        with torch.no_grad():
            y = X.float()
            self.location[0] = self.rate[0] = torch.mean(y).item()
            self.scale[0] =    torch.std(y)

        X1 = self.conv(X)

        with torch.no_grad():
            y = X1.float()
            self.location[1] = self.rate[1] = torch.mean(y).item()
            self.scale[1] =    torch.std(y)

        X1 = F.relu(self.bn(self.kernel1(X1) if ARGS.distribution else X1))
        if ARGS.debug: self.logger.debug(f"X1 shape: {X1.shape}")

        X2 = self.layer1(X1)
        if ARGS.debug: self.logger.debug(f"X2 shape: {X2.shape}")

        with torch.no_grad():
            y = X2.float()
            self.location[2] = self.rate[2] = torch.mean(y).item()
            self.scale[2] =    torch.std(y)

        X3 = self.layer2(X2)
        if ARGS.debug: self.logger.debug(f"X3 shape: {X3.shape}")

        with torch.no_grad():
            y = X3.float()
            self.location[3] = self.rate[3] = torch.mean(y).item()
            self.scale[3] =    torch.std(y)

        X4 = self.layer3(X3)
        if ARGS.debug: self.logger.debug(f"X4 shape: {X4.shape}")

        with torch.no_grad():
            y = X4.float()
            self.location[4] = self.rate[4] = torch.mean(y).item()
            self.scale[4] =    torch.std(y)

        X5 = F.avg_pool2d(self.layer4(X4), 4)

        with torch.no_grad():
            y = X5.float()
            self.location[5] = self.rate[5] = torch.mean(y).item()
            self.scale[5] =    torch.std(y)

        if self.training: self.record_params()

        return self.linear(X5.view(X5.size(0), -1))
    
    def record_params(self) -> None:
        """Record model parameters."""
        self.model_data.loc[len(self.model_data)] = [
            self.location[0], self.rate[0],
            self.location[1], self.rate[1],
            self.location[2], self.rate[2],
            self.location[3], self.rate[3],
            self.location[4], self.rate[4],
            self.location[5], self.rate[5],
        ]

    def update_kernels(self, epoch: int) -> None:
        """Update kernels.

        Args:
            epoch (int): Current epoch
        """
        self.kernel1 = get_kernel(ARGS.kernel_size, 64, location=self.location[1], scale=self.scale[1], rate=self.rate[1])

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for child in layer:
                child.update_kernels()

    def _initialize_weights(self) -> None:
        """Initialize weights in all layers."""
        for m in self.modules():

            # Convolving
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

            # Batch normalization
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

            # Linear
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes_out: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create ResNet block layer.

        Args:
            planes_out (int): Number of output planes
            num_blocks (int): Number of blocks within layer
            stride (int): Kernel convolution stride

        Returns:
            nn.Sequential: ResNetBlock layer
        """
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(ResnetBlock(self.planes_in, planes_out, stride))
            self.planes_in = planes_out * ResnetBlock.expansion

        return nn.Sequential(*layers)
    
    def to_csv(self, file_path: str) -> None:
        """Dump model data to CSV file.

        Args:
            file_path (str): Path to data file
        """
        self.model_data.to_csv(file_path)