"""Resnet-18 model."""

from logging                import Logger

from pandas                 import DataFrame
from torch                  import mean, no_grad, std, Tensor
from torch.cuda             import is_available
from torch.nn               import BatchNorm2d, Conv2d, Linear, Module, Sequential
from torch.nn.functional    import avg_pool2d, relu
from torch.nn.init          import constant_, kaiming_normal_, normal_

from kernels                import load_kernel
from models.resnet_block    import ResnetBlock
from utils                  import LOGGER

class Resnet(Module):
    """Resnet 18 model."""

    # Initialize lyer-data file
    _model_data = DataFrame(columns = [
        'Data-STD',     'Data-Mean',
        'Layer 1-STD',  'Layer 1-MEAN',
        'Layer 5-STD',  'Layer 5-MEAN',
        'Layer 9-STD',  'Layer 9-MEAN',
        'Layer 13-STD', 'Layer 13-MEAN',
        'Layer 18-STD', 'Layer 18-MEAN'
    ])

    def __init__(self, 
        channels_in:    int, 
        channels_out:   int,
        kernel:         str =   None,
        location:       float = 0.0,
        scale:          float = 1.0,
    ):
        """# Initialize Resnet 18 model.

        # Args:
            * channels_in   (int):              Input channels.
            * channels_out  (int):              Output channels.
            * kernel        (str, optional):    Kernel with which model will be set.
            * location      (float, optional):  Distribution location parameter. Defaults to 0.0.
            * scale         (float, optional):  Distribution scale parameter. Defaults to 1.0.
        """
        # Initialize Module object
        super(Resnet, self).__init__()

        # Initialize logger
        self.__logger__:    Logger =        LOGGER.getChild(suffix = 'resnet')

        # Initialize planes in
        self._planes_in_:   int =           64

        # Initialize distribution parameters
        self._kernel_:      str =           kernel
        self._locations_:   list[float] =   [location]*5
        self._scales_:      list[float] =   [scale]*5

        # Batch normalization layer
        self._bn:           BatchNorm2d =   BatchNorm2d(64)

        # Convolving layer
        self._conv_:        Conv2d =        Conv2d(channels_in, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Block layers
        self._layer1_:      Sequential =    self._make_layer_( 64, 2, stride=1)
        self._layer2_:      Sequential =    self._make_layer_(128, 2, stride=2)
        self._layer3_:      Sequential =    self._make_layer_(256, 2, stride=2)
        self._layer4_:      Sequential =    self._make_layer_(512, 2, stride=2)

        # Linear layer
        self._linear_:      Linear =        Linear(512, channels_out)

        # initialize layer weights
        self._initialize_weights_()

    def forward(self, 
        X:  Tensor
    ) -> Tensor:
        """# Feed input through network and produce output.

        ## Args:
            * X (Tensor):   Input tensor

        ## Returns:
            * Tensor:   Output tensor
        """
        # INPUT LAYER =============================================================================
        # Log input tensor shape for debugging
        self.__logger__.debug(f"Input layer shape: {X.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y = X.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[0], self._scales_[0] = mean(y).item(), std(y).item()

        # LAYER 1 =================================================================================
        # Pass through convolving layer
        x1:     Tensor =    self._conv_(X)

        # Log first layer output for debugging
        self.__logger__.debug(f"Layer 1 shape: {x1.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x1.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[1], self._scales_[1] = mean(y).item(), std(y).item()

        # LAYER 2 =================================================================================
        # Pass through first block layer
        x2:     Tensor =    self._layer1_(relu(self._bn_(self._kernel1_(x1) if self._kernel_ else x1)))

        # Log second layer output for debugging
        self.__logger__.debug(f"Layer 2 shape: {x2.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x2.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[2], self._scales_[2] = mean(y).item(), std(y).item()

        # LAYER 3 =================================================================================
        # Pass through second block layer
        x3:     Tensor =    self._layer2_(x2)

        # Log third layer output for debugging
        self.__logger__.debug(f"Layer 3 shape: {x3.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x3.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[3], self._scales_[3] = mean(y).item(), std(y).item()

        # LAYER 4 =================================================================================
        # Pass through third block layer
        x4:     Tensor =    self._layer3_(x3)

        # Log fourth layer output for debugging
        self.__logger__.debug(f"Layer 4 shape: {x4.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    x4.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[4], self._scales_[4] = mean(y).item(), std(y).item()

        # OUTPUT LAYER ============================================================================
        # Pass through fourth block layer
        output: Tensor =    avg_pool2d(self._layer4_(x4), 4)

        # Log output layer output for debugging
        self.__logger__.debug(f"Output layer shape: {output.shape}")

        # Without taking gradients...
        with no_grad(): 
            
            # Convert tensor to float values
            y:  Tensor =    output.float()
            
            # Calculate mean & standard deviation of layer output
            self._locations_[5], self._scales_[5] = mean(y).item(), std(y).item()

        # Return classification of sample
        return self._linear_(output.view(output.size(0), -1))
    
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

        # Set kernel
        self._kernel1_ = load_kernel(
                            kernel =    self._kernel_, 
                            size =      size, 
                            channels =  64, 
                            location =  self._locations_[1], 
                            scale =     self._scales_[1]
                        )

        # For each block layer...
        for layer in [self._layer1_, self._layer2_, self._layer3_, self._layer4_]:
            
            # For each block in block layer...
            for child in layer:
                
                # Set kernel(s) for block
                child.set_kernels(size = size)
            
        # Set model on GPU if available
        if is_available():  self = self.cuda()
    
    def record_params(self) -> None:
        """Record model distribution parameters."""
        self._model_data[len(self._model_data)] = [
            self._locations_[0], self.rate[0],
            self._locations_[1], self.rate[1],
            self._locations_[2], self.rate[2],
            self._locations_[3], self.rate[3],
            self._locations_[4], self.rate[4],
            self._locations_[5], self.rate[5],
        ]

    def _initialize_weights_(self) -> None:
        """Initialize weights in all layers."""
        # For each module...
        for module in self.modules():

            # For convolving layer(s)...
            if isinstance(obj =     module, class_or_tuple =    Conv2d):
                
                # Fill the input Tensor with values using a Kaiming normal distribution
                kaiming_normal_(tensor = module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None: constant_(module.bias, 0)

            # For batch normalization
            elif isinstance(obj =   module, class_or_tuple =    BatchNorm2d):
                
                # Fill the input Tensor with the value
                constant_(tensor =  module.weight,  val =   1)
                constant_(tensor =  module.bias,    val =   0)

            # Linear
            elif isinstance(obj =   module, class_or_tuple =    Linear):
                
                # Fill the input Tensor with values drawn from the normal distribution
                normal_(tensor = module.weight, mean = 0, std = 0.01)
                
                # Fill the input Tensor with the value
                constant_(tensor = module.bias, val = 0)

    def _make_layer_(self,
        planes_out: int,
        num_blocks: int,
        stride:     int
    ) -> Sequential:
        """# Create ResNet block layer.

        ## Args:
            * planes_out    (int):  Number of output planes
            * num_blocks    (int):  Number of blocks within layer
            * stride        (int):  Kernel convolution stride

        ## Returns:
            * Sequential:   ResNetBlock layer
        """
        # Calculate number of strides needed
        strides:                int =       [stride] + [1]*(num_blocks - 1)
        
        # Initialize list of layers
        layers:                 list[] =    []

        # For each stride amount needed...
        for stride in strides:
            
            # Append block layer
            layers.append(ResnetBlock(self._planes_in_, planes_out, stride))
            
            # "redefine" planes in
            self._planes_in_:   int =       planes_out

        # Return block layer
        return Sequential(*layers)
    
    def to_csv(self, file_path: str) -> None:
        """Dump model layer data to CSV file.

        Args:
            file_path (str): Path at which data file (CSV) will be written
        """
        self.__logger__.info(f"Saving model layer data to {file_path}")
        self._model_data.to_csv(file_path)