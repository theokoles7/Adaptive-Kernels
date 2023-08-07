"""Resnet model block."""

import torch, torch.nn as nn, torch.nn.functional as F

class ResnetBlock(nn.Module):

    def __init__(self, channels_in: int, channels_out: int, stride: int):
        """Initialize Resnet block.

        Args:
            channels_in (int): Number of input channels
            channels_out (int): Number of output channels
            stride (int): Kernel convolution stride
        """
        super(ResnetBlock, self).__init__()

        # Convolving layers
        self.conv1 =    nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 =    nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or channels_in != channels_out:
            self.shortcut_kernel = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels_out)
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Feed input through network and return output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        X = self.conv1(X)

        with torch.no_grad():
            y = X.float()
            self.std2 = torch.std(y).item()
            self.mean2 = torch.mean(y).item()

        X = self.conv2(F.relu(self.bn1(self.kernel1(X))))

        with torch.no_grad():
            y = X.float()
            self.std3 = torch.std(y).item()
            self.mean3 = torch.mean(y).item()

        X = self.bn2(self.kernel2(X))
        X += self.shortcut(X)

        return F.relu(X)