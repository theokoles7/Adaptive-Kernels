"""VGG 16 utilities."""

import pandas as pd, torch, torch.nn as nn

from kernels.accessor import get_kernel
from utils import ARGS, LOGGER

class VGG(nn.Module):
    """VGG 16 model."""

    # Initialize logger
    logger = LOGGER.getChild('vgg')

    # Initialize model data file
    model_data = pd.DataFrame(columns=[
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
        self.location =     [ARGS.location]*5
        self.scale =        [ARGS.scale]*5
        self.rate =         [ARGS.rate]*5

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
        """Feed input through network and return output.

        Args:
            X (torch.Tensor): Input tensor
            return_intermediate (bool, optional): Inidicate returning of intermediate or final output. Defaults to False.

        Returns:
            torch.Tensor: Output tensor
        """
        with torch.no_grad():
            y = X.float()
            self.location[0] = self.rate[0] = torch.mean(y).item()
            self.scale[0] = torch.std(y).item()

        X1 = self.conv1(X)

        with torch.no_grad():
            y = X1.float()
            self.location[1] = self.rate[1] = torch.mean(y).item()
            self.scale[1] = torch.std(y).item()

        X2 = self.conv2(self.pool1(self.kernel1(X1) if ARGS.distribution else X1))

        with torch.no_grad():
            y = X2.float()
            self.location[2] = self.rate[2] = torch.mean(y).item()
            self.scale[2] = torch.std(y).item()

        X3 = self.conv3(self.pool2(self.kernel2(X2) if ARGS.distribution else X2))

        with torch.no_grad():
            y = X3.float()
            self.location[3] = self.rate[3] = torch.mean(y).item()
            self.scale[3] = torch.std(y).item()

        X4 = self.conv4(self.pool3(self.kernel3(X3) if ARGS.distribution else X3))

        with torch.no_grad():
            y = X4.float()
            self.location[4] = self.rate[4] = torch.mean(y).item()
            self.scale[4] = torch.std(y).item()

        X5 = self.conv5(self.pool4(self.kernel4(X4) if ARGS.distribution else X4))

        with torch.no_grad():
            y = X5.float()
            self.location[5] = self.rate[5] = torch.mean(y).item()
            self.scale[5] = torch.std(y).item()

        if self.training: self.record_params()

        X5 = self.kernel5(X5) if ARGS.distribution else X5

        if return_intermediate: return X5.view(X5.size(0), -1)

        X5 = self.pool5(X5)

        return self.classifier(X5.view(X5.size(0), -1))
    
    def update_kernels(self, epoch: int) -> None:
        """_summary_

        Args:
            epoch (int): _description_
        """
        self.kernel1 = get_kernel(ARGS.kernel_size, channels= 64, location=self.location[0], scale=self.scale[0], rate=self.rate[0])
        self.kernel2 = get_kernel(ARGS.kernel_size, channels=128, location=self.location[1], scale=self.scale[1], rate=self.rate[1])
        self.kernel3 = get_kernel(ARGS.kernel_size, channels=256, location=self.location[2], scale=self.scale[2], rate=self.rate[2])
        self.kernel4 = get_kernel(ARGS.kernel_size, channels=512, location=self.location[3], scale=self.scale[3], rate=self.rate[3])
        self.kernel5 = get_kernel(ARGS.kernel_size, channels=512, location=self.location[4], scale=self.scale[4], rate=self.rate[4])

    def record_params(self) -> None:
        """Record model data."""
        self.model_data.loc[len(self.model_data)] = [
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
        """Dump model data to CSV file.

        Args:
            file_path (str): Path to data file
        """
        self.model_data.to_csv(file_path)