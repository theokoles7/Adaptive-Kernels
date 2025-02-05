"""Basic CNN model."""

from json                   import dump
from logging                import Logger

from pandas                 import DataFrame
from torch                  import mean, no_grad, std, Tensor
from torch.cuda             import is_available
from torch.nn               import Conv2d, Linear, MaxPool2d, Module
from torch.nn.functional    import relu

from kernels                import load_kernel
from utils                  import LOGGER

class NormalCNN(Module):
    """Basic CNN model."""

    # # Initialize layer-data file
    # _model_data =   DataFrame(columns = [
    #     'Data-STD',  'Layer 1-STD',  'Layer 2-STD',  'Layer 3-STD',  'Layer 4-STD',
    #     'Data-Mean', 'Layer 1-MEAN', 'Layer 2-MEAN', 'Layer 3-MEAN', 'Layer 4-MEAN'
    # ])

    def __init__(self,
        channels_in:    int, 
        channels_out:   int, 
        dim:            int,
        kernel:         str =   None,
        location:       float = 0.0,
        scale:          float = 1.0,
        **kwargs
    ):
        """# Initialize Normal CNN model.

        ## Args:
            * channels_in   (int):              Input channels.
            * channels_out  (int):              Output channels.
            * dim           (int):              Dimension of image (relevant for reshaping, post-convolution).
            * kernel        (str, optional):    Kernel with which model will be set.
            * location      (float, optional):  Distribution location parameter. Defaults to 0.0.
            * scale         (float, optional):  Distribution scale parameter. Defaults to 1.0.
        """
        # Initialize parent class
        super(NormalCNN, self).__init__()
    
        # Initialie model data record
        self._model_data_:  dict =  {}
        
        # Initialize logger
        self.__logger__:    Logger =    LOGGER.getChild(suffix = 'normal-cnn')

        # Initialize distribution parameters
        self._kernel_:      str =           kernel
        self._locations_:   list[float] =   [location]*5
        self._scales_:      list[float] =   [scale]*5

        # Convolving layers
        self._conv1_:       Conv2d =        Conv2d(channels_in,  32, kernel_size=3, padding=1)
        self._conv2_:       Conv2d =        Conv2d(         32,  64, kernel_size=3, padding=1)
        self._conv3_:       Conv2d =        Conv2d(         64, 128, kernel_size=3, padding=1)
        self._conv4_:       Conv2d =        Conv2d(        128, 256, kernel_size=3, padding=1)

        # Max pooling layers
        self._pool1_:       MaxPool2d =     MaxPool2d(kernel_size=2, stride=2)
        self._pool2_:       MaxPool2d =     MaxPool2d(kernel_size=2, stride=2)
        self._pool3_:       MaxPool2d =     MaxPool2d(kernel_size=2, stride=2)
        self._pool4_:       MaxPool2d =     MaxPool2d(kernel_size=2, stride=2)

        # FC layer
        self._fc_:          Linear =        Linear(dim**2, 1024)

        # Classifier
        self._classifier_:  Linear =        Linear(1024, channels_out)

    def forward(self,
        X:  Tensor
    ) -> Tensor:
        """# Feed input through network and produce output.

        ## Args:
            * X (Tensor):   Input tensor.

        ## Returns:
            * Tensor:   Output tensor.
        """
        # INPUT LAYER =============================================================================
        # Log input tensor shape for debugging
        self.__logger__.debug(f"Input shape: {X.shape}")

        # Without taking gradients...
        with no_grad():
            
            # Convert tensor to float values
            y:  Tensor =    X.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[0], self._scales_[0] = mean(y).item(), std(y).item()

        # LAYER 1 =================================================================================
        # Pass through first convolving layer
        x1:     Tensor =    self._conv1_(X)

        # Log first layer output for debugging
        self.__logger__.debug(f"Layer 1 output shape: {x1.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x1.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[1], self._scales_[1] = mean(y).item(), std(y).item()

        # LAYER 2 =================================================================================
        # Pass through second convolving layer
        x2:     Tensor =    self._conv2_(relu(self._pool1_(self._kernel1_(x1) if self._kernel_ is not None else x1)))

        # Log second layer output for debugging
        self.__logger__.debug(f"Layer 2 output shape: {x2.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x2.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[2], self._scales_[2] = mean(y).item(), std(y).item()

        # LAYER 3 =================================================================================
        # Pass through third convolving layer
        x3:     Tensor =    self._conv3_(relu(self._pool2_(self._kernel2_(x2) if self._kernel_ is not None else x2)))

        # Log third layer output for debugging
        self.__logger__.debug(f"Layer 3 output shape: {x3.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x3.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[3], self._scales_[3] = mean(y).item(), std(y).item()

        # LAYER 4 =================================================================================
        # Pass through third convolving layer
        x4:     Tensor =    self._conv4_(relu(self._pool3_(self._kernel3_(x3) if self._kernel_ is not None else x3)))

        # Log third layer output for debugging
        self.__logger__.debug(f"Layer 4 shape: {x4.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x4.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[4], self._scales_[4] = mean(y).item(), std(y).item()

        # OUTPUT LAYER ============================================================================
        output: Tensor =    relu(self._pool4_(self._kernel4_(x4) if self._kernel_ is not None else x4))

        self.__logger__.debug(f"Output shape: {output.shape}")

        # # Record parameters in data file if training
        # if self.training: self.record_params()

        # Return classified output
        return self._classifier_(relu(self._fc_(output.view(output.size(0), -1))))
    
    def set_kernels(self,
        epoch:      int,
        size:       int
    ) -> None:
        """# Create/update kernels.

        ## Args:
            * epoch (int):  Epoch during which kernels are being set.
            * size  (int):  Size with which kernels will be created.
        """
        # Log for debugging
        self.__logger__.debug(f"EPOCH {epoch} locations: {self._locations_}, scales: {self._scales_}")
        
        # Set current epoch
        self._epoch_:   int =   epoch

        # Set kernels
        for kernel, channel_size, location, scale in zip(
            ["_kernel1_", "_kernel2_", "_kernel3_", "_kernel4_"],
            [32, 64, 128, 256],
            self._locations_,
            self._scales_
        ):
            self.__setattr__(name = kernel, value = load_kernel(
                kernel =    self._kernel_,
                size =      size,
                channels =  channel_size,
                location =  location,
                scale =     scale
            ))
            
        # Set model on GPU if available
        if is_available():  self = self.cuda()

    def record_parameters(self) -> None:
        """# Record mean & standard deviation of layers in model data file."""
        # Record epoch parameters
        self._model_data_.update({
            self._epoch_:   {
                "location": self._locations_,
                "scale":    self._scales_
            }
        })

    def save_parameters(self, 
        file_path:  str
    ) -> None:
        """# Dump model layer data to CSV file.

        ## Args:
            * file_path (str):  Path at which data file (CSV) will be written
        """
        # Log action
        self.__logger__.info(f"Saving model layer data to {file_path}")
        
        # Save model data to file
        dump(
            obj =       self._model_data_,
            fp =        open(file = file_path, mode = "w"),
            indent =    2,
            default =   str
        )