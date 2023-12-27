"""Basic CNN model."""

import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

from kernels    import get_kernel
from utils      import ARGS, LOGGER

class NormalCNN(nn.Module):
    """Basic CNN model."""

    # Initialize logger
    _logger =       LOGGER.getChild('normal-cnn')

    # Initialize layer-data file
    _model_data =   pd.DataFrame(columns = [
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
        self.location =     [(ARGS.location if ARGS.distribution != "poisson" else ARGS.rate) if ARGS.distribution else 0.0]*5
        self.scale =        [ARGS.scale if ARGS.distribution else 1.0]*5

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
        """Feed input through network and produce output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # INPUT LAYER =============================================================================
        self._logger.debug(f"Input shape: {X.shape}")

        with torch.no_grad(): 
            y = X.float()
            self.location[0], self.scale[0] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 1 =================================================================================
        x1 =    self.conv1(X)

        self._logger.debug(f"Layer 1 shape: {x1.shape}")

        with torch.no_grad(): 
            y = x1.float()
            self.location[1], self.scale[1] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 2 =================================================================================
        x2 =    self.conv2(F.relu(self.pool1(self.kernel1(x1) if ARGS.distribution else x1)))

        self._logger.debug(f"Layer 2 shape: {x2.shape}")

        with torch.no_grad(): 
            y = x2.float()
            self.location[2], self.scale[2] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 3 =================================================================================
        x3 =    self.conv3(F.relu(self.pool2(self.kernel2(x2) if ARGS.distribution else x2)))

        self._logger.debug(f"Layer 3 shape: {x3.shape}")

        with torch.no_grad(): 
            y = x3.float()
            self.location[3], self.scale[3] = torch.mean(y).item(), torch.std(y).item()

        # LAYER 4 =================================================================================
        x4 =    self.conv4(F.relu(self.pool3(self.kernel3(x3) if ARGS.distribution else x3)))

        self._logger.debug(f"Layer 4 shape: {x4.shape}")

        with torch.no_grad(): 
            y = x4.float()
            self.location[4], self.scale[4] = torch.mean(y).item(), torch.std(y).item()

        # OUTPUT LAYER ============================================================================
        output =    F.relu(self.pool4(self.kernel4(x4) if ARGS.distribution else x4))

        self._logger.debug(f"Output shape: {output.shape}")

        # Record parameters in data file if training
        if self.training: self.record_params()

        # Return classified output
        return self.classifier(F.relu(self.fc(output.view(output.size(0), -1))))
    
    def set_kernels(self, epoch: int) -> None:
        """Create/update kernels.

        Args:
            epoch (int): Current epoch number
        """
        self._logger.debug(f"EPOCH {epoch} locations: {self.location}")
        self._logger.debug(f"EPOCH {epoch} scales:    {self.scale}")

        # Set kernels
        self.kernel1 = get_kernel(ARGS.distribution, ARGS.kernel_size,  32, location=self.location[0], scale=self.scale[0], rate=self.location[0])
        self.kernel2 = get_kernel(ARGS.distribution, ARGS.kernel_size,  64, location=self.location[1], scale=self.scale[1], rate=self.location[1])
        self.kernel3 = get_kernel(ARGS.distribution, ARGS.kernel_size, 128, location=self.location[2], scale=self.scale[2], rate=self.location[2])
        self.kernel4 = get_kernel(ARGS.distribution, ARGS.kernel_size, 256, location=self.location[3], scale=self.scale[3], rate=self.location[3])

    def record_params(self) -> None:
        """Record mean & standard deviation of layers in model data file."""
        self._model_data.loc[len(self._model_data)] = [
            self.location[0], self.location[1], self.location[2], self.location[3], self.location[4],
            self.scale[0],    self.scale[1],    self.scale[2],    self.scale[3],    self.scale[4]
        ]

    def to_csv(self, file_path: str) -> None:
        """Dump model layer data to CSV file.

        Args:
            file_path (str): Path at which data file (CSV) will be written
        """
        self._logger.info(f"Saving model layer data to {file_path}")
        self._model_data.to_csv(file_path)