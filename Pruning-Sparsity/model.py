import torch
import torch.nn as nn
import utils

class VGG16(nn.Module):
    """
    VGG11 model adapted for CIFAR-10 classification task or similar datasets with customizable input shape and number of classes.

    Attributes:
        num_classes (int): Number of output classes. Defaults to 10 for CIFAR-10.
        input_shape (tuple[int, int]): The height and width of the input images. Defaults to (32, 32) for CIFAR-10.
    """

    def __init__(self, 
                 num_classes: int = 10,
                 input_shape: tuple[int,int] = (32,32)):
        super(VGG16, self).__init__()
        """
        Initialize the VGG11 model with specified number of classes and input shape.

        Parameters:
            num_classes (int): Number of classes for the final classification layer.
            input_shape (tuple[int, int]): Resolution of the input images (height, width).
        """
        
        assert len(input_shape) == 2, "Give proper resolution of the input image."
        self.input_shape = input_shape
        
        self.features = self.create_continuous_features()
        
        # Get flattened shape for different input sizes.        
        flattened_shape = self.get_flattened_shape()
        
        self.classifier = nn.Sequential(
            nn.Linear(flattened_shape, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def create_continuous_features(self):
        """
        Creates the feature extractor part of the model based on VGG architecture.

        Returns:
            nn.Sequential: The feature extractor model.
        """
        # Define individual convolution blocks
        block1 = self.convolution_block(3,64)
        block2 = self.convolution_block(64,128)
        block3 = self.convolution_block(128,256)
        block4 = self.convolution_block(256,512)
        blocks = block1 + block2 + block3 + block4
        
        # Sequentially add convolution blocks to the features model
        # and name them.
        features = nn.Sequential()
        for count,layer in enumerate(blocks):
            if isinstance(layer,nn.Conv2d):
                layer_name = f'conv{utils.count_layer_instances(features,nn.Conv2d)}'
            if isinstance(layer,nn.ReLU):
                layer_name = f'relu{utils.count_layer_instances(features,nn.ReLU)}'
            if isinstance(layer,nn.MaxPool2d):
                layer_name = f'maxpool{utils.count_layer_instances(features,nn.MaxPool2d)}'
            if isinstance(layer,nn.BatchNorm2d):
                layer_name = f'batchnorm{utils.count_layer_instances(features,nn.BatchNorm2d)}'
            features.add_module(name=f'{layer_name}',module=layer)
            
        return features
    
    def convolution_block(self,
                          in_channels: int,
                          out_channels: int):
        """
        Creates a convolutional block consisting of Conv2D, BatchNorm2d, and ReLU layers.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            List[nn.Module]: A list containing the layers of the convolutional block.
        """
        block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        return block
    
    def get_flattened_shape(self):
        """
        Calculates the shape of the tensor after passing through the feature extractor.

        Returns:
            int: The number of features after flattening the output of the feature extractor.
        """
        dummy_input = torch.randn((1,3,self.input_shape[0],self.input_shape[1]))
        dummy_output = self.features(dummy_input)
        flattened_shape = torch.reshape(dummy_output,(1,-1)).shape
        return flattened_shape[1]  # Take the second value of the shape as flattened shape.
        
    def _initialize_weights(self):
        """
        Initializes the weights of the model with He initialization for Conv2d layers
        and normal initialization for Linear layers.
        """
        for m in self.modules():
            # Initialize weights with different methods.
            # Covered in report.
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


