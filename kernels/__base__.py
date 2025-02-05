"""Base implementation of kernel classes."""

from logging    import Logger
from random     import choice

from torch      import arange, index_select, isinf, LongTensor, stack, sum, Tensor
from torch.nn   import Conv2d

from utils      import LOGGER

class Kernel(Conv2d):
    """# Base Kernel class."""
    
    __kernel_groups__:  dict =  {
        1:  ['top-left',    'bottom-right'],
        2:  ['bottom-left', 'top-right'],
        3:  ['top-left',    'bottom-left'],
        4:  ['top-right',   'bottom-right'],
        5:  ['top-left',    'top-right'],
        6:  ['bottom-left', 'bottom-right'],
        7:  ['top-left',    'bottom-right', 'static'],
        8:  ['bottom-left', 'top-right',    'static'],
        9:  ['top-left',    'bottom-left',  'static'],
        10: ['top-right',   'bottom-right', 'static'],
        11: ['top-left',    'top-right',    'static'],
        12: ['bottom-left', 'bottom-right', 'static'],
        13: ['top-left',    'bottom-right', 'bottom-left', 'top-right'],
        14: ['top-left',    'bottom-right', 'bottom-left', 'top-right', 'static']
    }
    
    def __init__(self,
        kernel_group:   int =   13,
        location:       float = 0.0,
        scale:          float = 0.1,
        channels:       int =   3,
        size:           int =   3,
        **kwargs
    ):
        """# Initialize Kernel object.

        ## Args:
            * kernel_group  (int, optional):    Configuration group from which kernel type will be 
                                                randomly selected. Defaults to 13th group.
            * location      (float, optional):  Distribution location parameter.
            * scale         (float, optional):  Distribution scale parameter.
            * channels      (int, optional):    Input channels. Defaults to 3.
            * size          (int, optional):    Kernel size (square). Defaults to 3.
        """
        # Initialize Conv2d parent object
        super(Kernel, self).__init__(
            in_channels =   channels,
            out_channels =  channels,
            kernel_size =   size,
            groups =        channels,
            bias =          False,
            padding =       (1 if size == 3 else (2 if size == 5 else 0))
        )
        
        # Initialize location & scale attributes
        self._location_:    float =     location
        self._scale_:       float =     scale
        
        # Initialize logger
        self.__logger__:    Logger =    LOGGER.getChild(suffix = self.__class__.__name__.lower())
        
        # Randomly select kernel type from config group & log for debugging
        kernel_type:        str =       choice(self.__kernel_groups__[kernel_group])
        self.__logger__.debug(f"Kernel type selected: {kernel_type}")
        
        # Initialize Tensor seed
        seed:               Tensor =    arange(size)
        
        # Duplicate Tensor to have {size} copies (basically new Tensor produced here will be size x 
        # size x size)
        x_grid:             Tensor =    seed.repeat(size).view(size, size)
        
        # Record transpose
        y_grid:             Tensor =    x_grid.t()
        
        # Calculate kernel values
        kernel:             Tensor =    self.pdf(stack([x_grid, y_grid], dim = -1).float())
        
        # Ensure that kernel does not contain infinite values
        assert not isinf(kernel).any(), f"NAN or INF detected in kernel(s):\n{kernel}"
        
        # Normalize values ot be between 0 and 1
        kernel /= sum(kernel)
        
        # If the filter type selected is in list...
        if kernel_type in ["top-right", "bottom-left", "bottom-right"]:
            
            # If the filter type is bottom-right, specifically...
            if kernel_type == "bottom-right":
                
                # Swap first and last kernels
                temp:   Tensor =    kernel[0].clone()
                kernel[0] =         kernel[-1]
                kernel[-1] =        temp
                
            kernel: Tensor =    index_select(
                input = kernel,
                dim =   (0 if kernel_type == "bottom-left" else 1),
                index = LongTensor([2, 1, 0])
            )
            
        # Log kernel produced for debugging
        self.__logger__.debug(f"{kernel_type.upper()}:\n{kernel}")
        
        # Reshape kernel
        kernel:             Tensor =    kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)
        
        # Set kernel for Conv2d layer
        self.weight.data =              kernel
        
        # "Turn off" gradient updates
        self.weight.requires_grad =     False
        
    def pdf(self, **kwargs) -> Tensor:
        """# Calculate kernel values based on distribution of convolved layer output.

        ## Returns:
            * Tensor:   Kernel produced from probability distribution function.
        """
        raise NotImplementedError(f"Sub-class must override method.")