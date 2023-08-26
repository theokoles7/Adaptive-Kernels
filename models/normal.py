"""Basic CNN model."""

import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

from kernels.accessor import get_kernel
from utils.globals import *

class NormalCNN(nn.Module):
    """Basic CNN model"""

    model_data = pd.DataFrame(columns=['Data_Std',  'CNN_Layer_1_Std',  'CNN_Layer_2_Std',  'CNN_Layer_3_Std',  'CNN_Layer_4_Std',
                                       'Data_Mean', 'CNN_Layer_1_Mean', 'CNN_Layer_2_Mean', 'CNN_Layer_3_Mean', 'CNN_Layer_4_Mean'])

    def __init__(self, channels_in: int, channels_out: int, dim: int):
        """Initialize Normal CNN model.

        Args:
            channels_out (int): Input channels (Likely 1 if images are B&W, 3 if colored)
            channels_out (int): Output channels (number of classes in dataset)
            dim (int): Dimension of image (relevant for reshaping, post-convolution)
        """
        super(NormalCNN, self).__init__()

        # Initialize distribution parameters
        self.locations =    [ARGS.location]*4
        self.scales =       [ARGS.scale]*4
        self.rates =        [ARGS.rate]*4

        # Concolving layers
        self.conv1 = nn.Conv2d(  channels_in,  32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(           32,  64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(           64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(          128, 256, kernel_size=3, padding=1)

        # Max pooling layers
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc = nn.Linear(dim**2, 256 * 2 * 2)

        # Classifier
        self.classifier = nn.Linear(256 * 2 * 2, channels_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Feed input through network and return output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        if ARGS.debug: LOGGER.debug(f"Input shape: {X.shape}")

        x1 = self.conv1(X)

        if ARGS.debug: LOGGER.debug(f"x1 shape: {x1.shape}")

        with torch.no_grad():
            y = x1.float()
            self.rates[0] = torch.mean(y).item()
            self.locations[0], self.scales[0] = torch.mean(y).item(), torch.std(y).item()

        x2 = self.conv2(F.relu(self.max1(self.kernel1(x1) if ARGS.distribution else x1)))

        if ARGS.debug: LOGGER.debug(f"x2 shape: {x2.shape}")

        with torch.no_grad():
            y = x2.float()
            self.rates[1] = torch.mean(y).item()
            self.locations[1], self.scales[1] = torch.mean(y).item(), torch.std(y).item()

        x3 = self.conv3(F.relu(self.max2(self.kernel2(x2) if ARGS.distribution else x2)))

        if ARGS.debug: LOGGER.debug(f"x3 shape: {x3.shape}")

        with torch.no_grad():
            y = x3.float()
            self.rates[2] = torch.mean(y).item()
            self.locations[2], self.scales[2] = torch.mean(y).item(), torch.std(y).item()

        x4 = self.conv4(F.relu(self.max3(self.kernel3(x3) if ARGS.distribution else x3)))

        if ARGS.debug: LOGGER.debug(f"x4 shape: {x4.shape}")

        with torch.no_grad():
            y = x3.float()
            self.rates[3] = torch.mean(y).item()
            self.locations[3], self.scales[3] = torch.mean(y).item(), torch.std(y).item()

        x4 = F.relu(self.max4(self.kernel4(x4) if ARGS.distribution else x4))

        if self.training: self.record_params(X, x1, x2, x3, x4)

        return self.classifier(F.relu(self.fc(x4.view(x4.size(0), -1))))

    def update_kernels(self, epoch: int) -> None:
        """Update kernels and decay scale/rate if neessary.

        Args:
            epoch (int): Current epoch
        """
        if ARGS.debug:
            LOGGER.debug(f"EPOCH {epoch} locations: {self.locations}")
            LOGGER.debug(f"EPOCH {epoch} scales: {self.scales}")

        # Update kernels
        # @NOTE: Though all 3 parameters are being passed to the kernel accessor, only one set 
        # (univariate/bivariate) is actually going to be used to initialize the new kernels.
        self.kernel1 = get_kernel(location=self.locations[0], scale=self.scales[0], rate=self.rates[0], channels= 32)
        self.kernel2 = get_kernel(location=self.locations[1], scale=self.scales[1], rate=self.rates[1], channels= 64)
        self.kernel3 = get_kernel(location=self.locations[2], scale=self.scales[2], rate=self.rates[2], channels=128)
        self.kernel4 = get_kernel(location=self.locations[3], scale=self.scales[3], rate=self.rates[3], channels=256)

    def record_params(self, x, x1, x2, x3, x4):
        """Record model's location and scale parameters.

        Args:
            x  (torch.Tensor): Intermediate tensor
            x1 (torch.Tensor): Intermediate tensor
            x2 (torch.Tensor): Intermediate tensor
            x3 (torch.Tensor): Intermediate tensor
            x4 (torch.Tensor): Intermediate tensor
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
                round(torch.std(  x4.float()).item(), 3)
            ]

    def setTrainingModelStatus(self,modelStatus):
        self.modelStatus = modelStatus

    def to_csv(self, file_path: str) -> None:
        """Dump all data to CSV file.

        Args:
            file_path (str): Path at which model data will be saved
        """
        self.model_data.to_csv(file_path)