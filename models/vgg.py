"""VGG16 model."""

import pandas as pd
import torch, torch.nn as nn

from kernels.accessor import get_kernel
from utils.globals import ARGS

class VGG(nn.Module):
    """VGG16 model."""

    model_data = pd.DataFrame(columns=[
        'Data Mean', 'Data STD',
        'Layer 3 Mean', 'Layer 3 STD',
        'Layer 6 Mean', 'Layer 6 STD',
        'Layer 9 Mean', 'Layer 9 STD',
        'Layer 12 Mean', 'Layer 12 STD',
        'Layer 16 Mean', 'Layer 16 STD'
    ])

    std1 = std2 = std3 = std4 = std5 = 1
    mean1 = mean2 = mean3 = mean4 = mean5 = 1

    def __init__(self, channels_in: int, channels_out: int):
        """Initialize VGG model.

        Args:
            channels_out (int): Input channels (Likely 1 if images are B&W, 3 if colored)
            channels_out (int): Output channels (number of classes in dataset)
        """
        super(VGG, self).__init__()

        # Initialize distribution parameters
        self.locations =    [ARGS.location]*5
        self.scales =       [ARGS.scale]*5
        self.rates =        [ARGS.rate]*5

        # Convolving layers
        self.conv1 = nn.Sequential(nn.Conv2d(channels_in,  64, 3, padding=1), nn.ReLU(), nn.Conv2d( 64,  64, 3, padding=1))
        self.conv2 = nn.Sequential(nn.Conv2d(         64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1))
        self.conv3 = nn.Sequential(nn.Conv2d(        128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1))
        self.conv4 = nn.Sequential(nn.Conv2d(        256, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1))
        self.conv5 = nn.Sequential(nn.Conv2d(        512, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1))

        # Pooling layers
        self.pool1 = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2, stride=2))
        self.pool2 = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2, stride=2))
        self.pool3 = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2, stride=2))
        self.pool4 = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2, stride=2))
        self.pool5 = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2, stride=2))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear( 512, 4096), nn.ReLU(), torch.nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), torch.nn.Dropout(),
            nn.Linear(4096, channels_out)
        )
        
        self._initialize_weights()

    def forward(self, X: torch.Tensor, return_intermediate: bool = False) -> torch.Tensor:
        """Feed input through network and return output.

        Args:
            X (torch.Tensor): Input tensor
            return_intermediate (bool, optional): Indicate returning of intermediate or final output. Defaults to False.

        Returns:
            torch.Tensor: Output tensor or sequential layer
        """
        x1 = self.conv1(X)
        with torch.no_grad():
         y = x1.float()
         self.scales[0] = torch.std(y).item()
         self.locations[0] = torch.mean(y).item()
         self.rates[0] = torch.mean(y).item()
         
        x1 = self.pool1(self.kernel1(x1) if ARGS.distribution else x1)

        x2 = self.conv2(x1)
        with torch.no_grad():
         y = x2.float()
         self.scales[1] = torch.std(y).item()
         self.locations[1] = torch.mean(y).item()
         self.rates[1] = torch.mean(y).item()
         
        x2 = self.pool2(self.kernel2(x2) if ARGS.distribution else x2)

        x3 = self.conv3(x2)
        with torch.no_grad():
         y = x3.float()
         self.scales[2] = torch.std(y).item()
         self.locations[2] = torch.mean(y).item()
         self.rates[2] = torch.mean(y).item()
         
        x3 = self.pool3(self.kernel3(x3) if ARGS.distribution else x3)

        x4 = self.conv4(x3)
        with torch.no_grad():
         y = x4.float()
         self.scales[3] = torch.std(y).item()
         self.locations[3] = torch.mean(y).item()
         self.rates[3] = torch.mean(y).item()
         
        x4 = self.pool4(self.kernel4(x4) if ARGS.distribution else x4)

        x5 = self.conv5(x4)
        with torch.no_grad():
         y = x5.float()
         self.scales[4] = torch.std(y).item()
         self.locations[4] = torch.mean(y).item()
         self.rates[4] = torch.mean(y).item()
         
        x5 = self.kernel5(x5) if ARGS.distribution else x5

        if return_intermediate:
            return x5.view(x5.size(0), -1)

        x5 = self.pool5(x5)

        if self.training: self.record_params(X, x1, x2, x3, x4, x5)

        return self.classifier(x5.view(x5.size(0), -1))
    
    def update_kernels(self, epoch: int) -> None:
        """Update kernels.

        Args:
            epoch (int): Current epoch
        """
        if epoch % 5 == 0: 
            if ARGS.distribution != 'poisson':
                ARGS.scale *= 0.9
            else:
                ARGS.rate *= 0.9

        self.kernel1 = get_kernel(location=self.locations[0], scale=self.scales[0], rate=self.rates[0], channels=64)
        self.kernel2 = get_kernel(location=self.locations[1], scale=self.scales[1], rate=self.rates[1], channels=128)
        self.kernel3 = get_kernel(location=self.locations[2], scale=self.scales[2], rate=self.rates[2], channels=256)
        self.kernel4 = get_kernel(location=self.locations[3], scale=self.scales[3], rate=self.rates[3], channels=512)
        self.kernel5 = get_kernel(location=self.locations[4], scale=self.scales[4], rate=self.rates[4], channels=512)

    def record_params(self, x, x1, x2, x3, x4, x5):
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
                round(torch.std(  x4.float()).item(), 3),
                round(torch.mean( x5.float()).item(), 3),
                round(torch.std(  x5.float()).item(), 3)
            ]

    def _initialize_weights(self) -> None:
        """Initialize weights of model layers.
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

    def to_csv(self, file_path: str) -> None:
        """Dump all data to CSV file.

        Args:
            file_path (str): Path at which model data will be saved
        """
        self.model_data.to_csv(file_path)