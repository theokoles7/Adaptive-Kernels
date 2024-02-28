"""VGG 16 utilities."""

import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

from kernels    import get_kernel
from utils      import ARGS, LOGGER

class VGG(nn.Module):
    """VGG 16 model."""

    # Initialize logger
    _logger = LOGGER.getChild('vgg')

    # Initializemodel data file
    _model_data = pd.DataFrame(columns=[
        'Input Mean',    'Input STD',
        'Layer 3-MEAN',  'Layer 3-STD',
        'Layer 6-MEAN',  'Layer 6-STD',
        'Layer 9-MEAN',  'Layer 9-STD',
        'Layer 12-MEAN', 'Layer 12-STD',
        'Layer 16-MEAN', 'Layer 16-STD',
    ])

    def __init__(self, channels_in: int, channels_out: int):
        """Initialize VGG 16 model.

        Args:
            channels_in (int): Input channels
            channels_out (int): Output channels
        """
        super(VGG, self).__init__()

        # Initialize distribution parameters
        self.location =     [(ARGS.location if ARGS.distribution != "poisson" else ARGS.rate) if ARGS.distribution else 0.0]*6
        self.scale =        [ARGS.scale if ARGS.distribution else 1.0]*6

        # Convolving layers
        self.conv1 =        nn.Sequential(nn.Conv2d(channels_in,  64, 3, padding=1), nn.ReLU(), nn.Conv2d(  64,  64, 3, padding=1))
        self.conv2 =        nn.Sequential(nn.Conv2d(         64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d( 128, 128, 3, padding=1))
        self.conv3 =        nn.Sequential(nn.Conv2d(        128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d( 256, 256, 3, padding=1))
        self.conv4 =        nn.Sequential(nn.Conv2d(        256, 512, 3, padding=1), nn.ReLU(), nn.Conv2d( 512, 512, 3, padding=1))
        self.conv5 =        nn.Sequential(nn.Conv2d(        512, 512, 3, padding=1), nn.ReLU(), nn.Conv2d( 512, 512, 3, padding=1))

        # Pooling layers
        self.pool1 =        nn.Sequential(nn.ReLU(), nn.MaxPool2d(2, stride=2))
        self.pool2 =        nn.Sequential(nn.ReLU(), nn.MaxPool2d(2, stride=2))
        self.pool3 =        nn.Sequential(nn.ReLU(), nn.MaxPool2d(2, stride=2))
        self.pool4 =        nn.Sequential(nn.ReLU(), nn.MaxPool2d(2, stride=2))
        self.pool5 =        nn.Sequential(nn.ReLU(), nn.MaxPool2d(2, stride=2))

        # Classifier
        self.classifier =   nn.Sequential(
            nn.Linear( 512, 4096), nn.ReLU(), torch.nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), torch.nn.Dropout(),
            nn.Linear(4096, channels_out)
        )

        self._initialize_weights()

    def forward(self, X: torch.Tensor, return_intermediate: bool = False) -> torch.Tensor:
        """Feed input through network and provide output.

        Args:
            X (torch.Tensor): Input tensor
            return_intermediate (bool, optional): If True, returns intermediate output, prior to classification. Defaults to False.

        Returns:
            torch.Tensor: Output tensor
        """
        # INPUT LAYER =============================================================================
        self._logger.debug(f"Input layer shape: {X.shape}")

        with torch.no_grad(): 
            y = X.float()
            self.location[0], self.scale[0] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 1 =================================================================================
        x1 = self.conv1(X)

        self._logger.debug(f"Layer 1 shape: {x1.shape}")

        with torch.no_grad(): 
            y = x1.float()
            self.location[1], self.scale[1] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 2 =================================================================================
        x2 = self.conv2(self.pool1(self.kernel1(x1) if ARGS.distribution else x1))

        self._logger.debug(f"Layer 2 shape: {x2.shape}")

        with torch.no_grad(): 
            y = x2.float()
            self.location[2], self.scale[2] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 3 =================================================================================
        x3 = self.conv3(self.pool2(self.kernel1(x2) if ARGS.distribution else x2))

        self._logger.debug(f"Layer 3 shape: {x3.shape}")

        with torch.no_grad(): 
            y = x3.float()
            self.location[3], self.scale[3] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 4 =================================================================================
        x4 = self.conv4(self.pool3(self.kernel1(x3) if ARGS.distribution else x3))

        self._logger.debug(f"Layer 4 shape: {x4.shape}")

        with torch.no_grad(): 
            y = x4.float()
            self.location[4], self.scale[4] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 5 =================================================================================
        x5 = self.conv5(self.pool4(self.kernel1(x4) if ARGS.distribution else x4))

        self._logger.debug(f"Layer 5 shape: {x5.shape}")

        with torch.no_grad(): 
            y = x5.float()
            self.location[5], self.scale[5] = torch.mean(y).item(), torch.std(y).item()

        # OUTPUT LAYER ============================================================================
        # Record model distribution parameters
        if self.training: self.record_params()

        output = self.pool5(self.kernel5(x5) if ARGS.distribution else x5)
        return self.classifier(output.view(output.size(0), -1))
    
    def set_kernels(self, epoch: int) -> None:
        """Create/update kernels.

        Args:
            epoch (int): Current epoch number
        """
        self.kernel1 = get_kernel(ARGS.distribution, ARGS.kernel_size,  64, self.location[0], self.scale[0], self.location[0])
        self.kernel2 = get_kernel(ARGS.distribution, ARGS.kernel_size, 128, self.location[1], self.scale[1], self.location[1])
        self.kernel3 = get_kernel(ARGS.distribution, ARGS.kernel_size, 256, self.location[2], self.scale[2], self.location[2])
        self.kernel4 = get_kernel(ARGS.distribution, ARGS.kernel_size, 512, self.location[3], self.scale[3], self.location[3])
        self.kernel5 = get_kernel(ARGS.distribution, ARGS.kernel_size, 512, self.location[4], self.scale[4], self.location[4])

    def record_params(self) -> None:
        """Record model distribution parameters."""
        self._model_data.loc[len(self._model_data)] = [
            self.location[0], self.scale[0],
            self.location[1], self.scale[1],
            self.location[2], self.scale[2],
            self.location[3], self.scale[3],
            self.location[4], self.scale[4],
            self.location[5], self.scale[5],
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

    def to_csv(self, file_path: str) -> None:
        """Dump model layer data to CSV file.

        Args:
            file_path (str): Path at which data file (CSV) will be written
        """
        self._logger.info(f"Saving model layer data to {file_path}")
        self._model_data.to_csv(file_path)