"""Basic CNN model."""

import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

from kernels.accessor import get_kernel
from utils import ARGS, LOGGER

class NormalCNN(nn.Module):
    """Basic CNN model."""

    # Initialize logger
    logger = LOGGER.getChild('normal-cnn')

    # Initialize layer data file
    model_data = pd.DataFrame(columns=[
        'Data-STD',  'Layer 1-STD',  'Layer 2-STD',  'Layer 3-STD',  'Layer 4-STD',
        'Data-Mean', 'Layer 1-MEAN', 'Layer 2-MEAN', 'Layer 3-MEAN', 'Layer 4-MEAN'
    ])

    def __init__(self, channels_in: int, channels_out: int, dim: int):
        """Initialize Normal CNN model.

        Args:
            channels_in (int): Input channels
            channels_out (int): Output channels
            dim (int): Dimension of image (relevant for reshaping, post-convolution)
        """
        super(NormalCNN, self).__init__()

        # Initialize distribution parameters
        self.location =     [ARGS.location]*5
        self.scale =        [ARGS.scale]*5
        self.rate =         [ARGS.rate]*5

        # Convolving layers
        self.conv1 =        nn.Conv2d(  channels_in,  32, kernel_size=3, padding=1)
        self.conv2 =        nn.Conv2d(           32,  64, kernel_size=3, padding=1)
        self.conv3 =        nn.Conv2d(           64, 128, kernel_size=3, padding=1)
        self.conv4 =        nn.Conv2d(          128, 256, kernel_size=3, padding=1)

        # Max pooling layers
        self.pool1 =        nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 =        nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 =        nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 =        nn.MaxPool2d(kernel_size=2, stride=2)

        # FC layer
        self.fc =           nn.Linear(dim**2, 1024)

        # Classifier
        self.classifier =   nn.Linear(1024, channels_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Feed input through network and return output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        if ARGS.debug: self.logger.debug(f"Input shape: {X.shape}")

        with torch.no_grad():
            y = x1.float()
            self.location[0] =  torch.mean(y).item()
            self.scale[0] =     torch.std(y).item()
            self.rate[0] =      torch.mean(y).item()

        x1 = self.conv1(X)

        with torch.no_grad():
            y = x1.float()
            self.location[1] =  torch.mean(y).item()
            self.scale[1] =     torch.std(y).item()
            self.rate[1] =      torch.mean(y).item()

        if ARGS.debug: self.logger.debug(f"X1 shape: {x1.shape}")

        x2 = self.conv2(F.relu(self.pool1(self.kernel1(x1) if ARGS.distribution else x1)))

        with torch.no_grad():
            y = x2.float()
            self.location[2] =  torch.mean(y).item()
            self.scale[2] =     torch.std(y).item()
            self.rate[2] =      torch.mean(y).item()

        if ARGS.debug: self.logger.debug(f"X2 shape: {x2.shape}")

        x3 = self.conv3(F.relu(self.pool2(self.kernel2(x2) if ARGS.distribution else x2)))

        with torch.no_grad():
            y = x3.float()
            self.location[3] =  torch.mean(y).item()
            self.scale[3] =     torch.std(y).item()
            self.rate[3] =      torch.mean(y).item()

        if ARGS.debug: self.logger.debug(f"X3 shape: {x3.shape}")

        x4 = self.conv4(F.relu(self.pool3(self.kernel3(x3) if ARGS.distribution else x3)))

        with torch.no_grad():
            y = x4.float()
            self.location[4] =  torch.mean(y).item()
            self.scale[4] =     torch.std(y).item()
            self.rate[4] =      torch.mean(y).item()

        if ARGS.debug: self.logger.debug(f"X4 shape: {x4.shape}")

        x5 = F.relu(self.pool4(self.kernel4(x4) if ARGS.distribution else x4))

        if ARGS.debug: self.logger.debug(f"X5 shape: {x5.shape}")

        if self.training: self.record_params(X, x1, x2, x3, x4)

        return self.classifier(F.relu(self.fc(x5.view(x5.size(0), -1))))
    
    def update_kernels(self, epoch: int) -> None:
        """Update kernels.

        Args:
            epoch (int): Current epoch
        """
        if ARGS.debug:
            self.logger.debug(f"EPOCH {epoch} locations: {self.location}")
            self.logger.debug(f"EPOCH {epoch} scales:    {self.scale}")
            self.logger.debug(f"EPOCH {epoch} rates:     {self.rate}")

        # Update kernels
        self.kernel1 = get_kernel(ARGS.kernel_size, channels= 32, location=self.location[0], scale=self.scale[0], rate=self.rate[0])
        self.kernel2 = get_kernel(ARGS.kernel_size, channels= 64, location=self.location[1], scale=self.scale[1], rate=self.rate[1])
        self.kernel3 = get_kernel(ARGS.kernel_size, channels=128, location=self.location[2], scale=self.scale[2], rate=self.rate[2])
        self.kernel4 = get_kernel(ARGS.kernel_size, channels=256, location=self.location[3], scale=self.scale[3], rate=self.rate[3])
                
    def record_params(self) -> None:
        """Record mean and standard deviation of layers into model data file.
        """
        self.model_data.loc[len(self.model_data)] = [
            self.location[0], self.location[1], self.location[2], self.location[3], self.location[4],
            self.scale[0],    self.scale[1],    self.scale[2],    self.scale[3],    self.scale[4]
        ]

    def to_csv(self, file_path: str) -> None:
        """Dump model data to CSV file.

        Args:
            file_path (str): Path to data file
        """
        self.model_data.to_csv(file_path)