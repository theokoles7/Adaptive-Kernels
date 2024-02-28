"""Resnet 18 model."""

import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

from kernels    import get_kernel
from utils      import ARGS, LOGGER
from models     import ResnetBlock

class Resnet(nn.Module):
    """Resnet 18 model."""

    # Initialize logger
    _logger = LOGGER.getChild('resnet')

    # Initialize lyer-data file
    _model_data = pd.DataFrame(columns = [
        'Data-STD',     'Data-Mean',
        'Layer 1-STD',  'Layer 1-MEAN',
        'Layer 5-STD',  'Layer 5-MEAN',
        'Layer 9-STD',  'Layer 9-MEAN',
        'Layer 13-STD', 'Layer 13-MEAN',
        'Layer 18-STD', 'Layer 18-MEAN'
    ])

    def __init__(self, channels_in: int, channels_out: int):
        """Initialize Resnet 18 model.

        Args:
            channels_in (int): Input channels
            channels_out (int): Output channels
        """
        super(Resnet, self).__init__()

        # Initialize planes in
        self._planes_in =  64

        # Initialize distribution parameters
        self.location =     [(ARGS.location if ARGS.distribution != "poisson" else ARGS.rate) if ARGS.distribution else 0.0]*6
        self.scale =        [ARGS.scale if ARGS.distribution else 1.0]*6

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
        self.linear =       nn.Linear(512, channels_out)

        self._initialize_weights()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Feed input through network and produce output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # INPUT LAYER =============================================================================
        self._logger.debug(f"Input layer shape: {X.shape}")

        with torch.no_grad(): 
            y = X.float()
            self.location[0], self.scale[0] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 1 =================================================================================
        x1 = self.conv(X)

        self._logger.debug(f"Layer 1 shape: {x1.shape}")

        with torch.no_grad(): 
            y = x1.float()
            self.location[1], self.scale[1] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 2 =================================================================================
        x2 = self.layer1(F.relu(self.bn(self.kernel1(x1) if ARGS.distribution else x1)))

        self._logger.debug(f"Layer 2 shape: {x2.shape}")

        with torch.no_grad(): 
            y = x2.float()
            self.location[2], self.scale[2] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 3 =================================================================================
        x3 = self.layer2(x2)

        self._logger.debug(f"Layer 3 shape: {x3.shape}")

        with torch.no_grad(): 
            y = x3.float()
            self.location[3], self.scale[3] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 4 =================================================================================
        x4 = self.layer3(x3)

        self._logger.debug(f"Layer 4 shape: {x4.shape}")

        with torch.no_grad(): 
            y = x4.float()
            self.location[4], self.scale[4] = torch.mean(y).item(), torch.std(y).item()

        # OUTPUT LAYER ============================================================================
        output = F.avg_pool2d(self.layer4(x4), 4)

        self._logger.debug(f"Output layer shape: {output.shape}")

        with torch.no_grad(): 
            y = output.float()
            self.location[5], self.scale[5] = torch.mean(y).item(), torch.std(y).item()

        return self.linear(output.view(output.size(0), -1))
    
    def set_kernels(self, epoch: int) -> None:
        """Create/update kernels.

        Args:
            epoch (int): Current epoch number
        """
        self.kernel1 = get_kernel(ARGS.distribution, ARGS.kernel_size, 64, self.location[1], self.scale[1], self.location[1])

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for child in layer:
                child.set_kernels()
    
    def record_params(self) -> None:
        """Record model distribution parameters."""
        self._model_data[len(self._model_data)] = [
            self.location[0], self.rate[0],
            self.location[1], self.rate[1],
            self.location[2], self.rate[2],
            self.location[3], self.rate[3],
            self.location[4], self.rate[4],
            self.location[5], self.rate[5],
        ]

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
            layers.append(ResnetBlock(self._planes_in, planes_out, stride))
            self._planes_in = planes_out

        return nn.Sequential(*layers)
    
    def to_csv(self, file_path: str) -> None:
        """Dump model layer data to CSV file.

        Args:
            file_path (str): Path at which data file (CSV) will be written
        """
        self._logger.info(f"Saving model layer data to {file_path}")
        self._model_data.to_csv(file_path)