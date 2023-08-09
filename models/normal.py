"""Basic CNN model."""

import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

from kernels.accessor import get_kernel
from utils.globals import *

class NormalCNN(nn.Module):
    """Basic CNN model"""

    model_data = pd.DataFrame(columns=['Data_Std',  'CNN_Layer_1_Std',  'CNN_Layer_2_Std',  'CNN_Layer_3_Std',  'CNN_Layer_4_Std',
                                       'Data_Mean', 'CNN_Layer_1_Mean', 'CNN_Layer_2_Mean', 'CNN_Layer_3_Mean', 'CNN_Layer_4_Mean'])

    def __init__(self, channels_in: int, channels_out: int):
        """Initialize Normal CNN model.

        Args:
            channels_out (int): Input channels (Likely 1 if images are B&W, 3 if colored)
            channels_out (int): Output channels (number of classes in dataset)
        """
        super(NormalCNN, self).__init__()

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
        self.fc = nn.Linear(256 * 2 * 2, 256 * 2 * 2)

        # Classifier
        self.classifier = nn.Linear(256 * 2 * 2, channels_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Feed input through network and return output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x1 = F.relu(self.max1(self.conv1(X)))
        x2 = F.relu(self.max2(self.conv2(x1)))
        x3 = F.relu(self.max3(self.conv3(x2)))
        x4 = F.relu(self.max4(self.conv4(x3)))

        if self.training: self.record_params(X, x1, x2, x3, x4)

        return self.classifier(F.relu(self.fc(x4.view(x4.size(0), -1))))

    def update_kernels(self, epoch: int) -> None:
        """Update kernels and decay scale/rate if neessary.

        Args:
            epoch (int): Current epoch
        """
        # For every 10 epochs, administer decay
        if epoch % 10 == 0:
            if ARGS.distribution != 'poisson':
                ARGS.scale *= 0.925
            else:
                ARGS.rate *= 0.925

        # Update kernels
        self.kernel0 = get_kernel(channels=  3)
        self.kernel1 = get_kernel(channels= 32)
        self.kernel2 = get_kernel(channels= 64)
        self.kernel3 = get_kernel(channels=128)
        self.kernel4 = get_kernel(channels=256)

    def record_params(self, x, x1, x2, x3, x4):
        """Record model's location and scale parameters.

        Args:
            x (torch.Tensor): Intermediate tensor
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