"""VGG16 model."""

import torch, torch.nn as nn

class VGG(nn.Module):
    """VGG16 model."""

    def __init__(self, channels_out: int):
        """Initialize VGG model.

        Args:
            channels_out (int): Output channels
        """
        super(VGG, self).__init__()

        # Convolving layers
        self.conv1 = nn.Sequential(nn.Conv2d(  3,  64, 3, padding=1), nn.ReLU(), nn.Conv2d( 64,  64, 3, padding=1))
        self.conv2 = nn.Sequential(nn.Conv2d( 64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1))

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
         self.std1 = torch.std(y).item()
         self.mean1 = torch.mean(y).item()
         
        x1 = self.post1(self.kernel1(x1))

        x2 = self.conv2(x1)
        with torch.no_grad():
         y = x2.float()
         self.std2 = torch.std(y).item()
         self.mean2 = torch.mean(y).item()
         
        x2 = self.post2(self.kernel2(x2))

        x3 = self.conv3(x2)
        with torch.no_grad():
         y = x3.float()
         self.std3 = torch.std(y).item()
         self.mean3 = torch.mean(y).item()
         
        x3 = self.post3(self.kernel3(x3))

        x4 = self.conv4(x3)
        with torch.no_grad():
         y = x4.float()
         self.std4 = torch.std(y).item()
         self.mean4 = torch.mean(y).item()
         
        x4 = self.post4(self.kernel4(x4))

        x5 = self.conv5(x4)
        with torch.no_grad():
         y = x5.float()
         self.std5 = torch.std(y).item()
         self.mean5 = torch.mean(y).item()
         
        x5 = self.kernel5(x5)

        if return_intermediate:
            return x5.view(x5.size(0), -1)

        x5 = self.post5(x5)

        if self.modelStatus:
            self.get_std(X, x1, x2, x3, x4, x5)

        return self.classifier(x5.view(x5.size(0), -1))

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