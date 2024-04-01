import torch
import torch.nn as nn

class Pruner():
    """
    Base class for implementing pruning strategies on PyTorch models.

    Attributes:
        module (nn.Module): The model or a specific module from a PyTorch model to be pruned.
        sparsity_levels (dict): A tuple of sparsity levels corresponding to each layer in the module.
        
    """
    def __init__(self,
                 module: nn.Module,
                 sparsity_levels: dict) -> None:
        """
        Initializes the Pruner object with a model/module and corresponding sparsity levels.

        Parameters:
            module (nn.Module): The model or module to apply pruning to.
            sparsity_levels (dict): Dictionary of desired sparsity values as values and layer names as keys.
        """
        self.module = module
        self.sparsity_levels = sparsity_levels
        
    def prune_layer(self,
                    layer: nn.Conv2d,
                    sparsity: float):
        """
        Placeholder for the prune_layer method to be implemented by subclasses.

        Parameters:
            layer (nn.Module): The layer to prune.
            sparsity (float): The desired sparsity level for the layer.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_masks(self) -> dict:
        """
        Prunes and generates pruning masks for each layer in the module based on the specified sparsity levels.

        Returns:
            dict: A dictionary mapping each layer to its corresponding pruning mask.
        """
        # Dictionary with masks of the modules.
        self.masks_dict = {}
        # Loop over the module.
        for key in self.sparsity_levels:
            layer = self.module._modules[key]
            mask = self.prune_layer(layer,self.sparsity_levels[key])
            self.masks_dict.update({key:mask})
                
        return self.masks_dict


class PrunerFinegrained(Pruner):
    """
    Implements fine-grained pruning, targeting individual weights in convolutional layers.
    """
    def __init__(self,
                 module: nn.Module,
                 sparsity_levels: tuple[float]) -> None:
        """
        Initializes the fine-grained pruner with a model/module and sparsity levels.

        Parameters:
            module (nn.Module): The model or module to apply fine-grained pruning to.
            sparsity_levels (tuple[float]): The desired sparsity levels for each layer in the module.
        """
        super(PrunerFinegrained,self).__init__(module,sparsity_levels)
        
    def prune_layer(self,
                    layer: nn.Conv2d,
                    sparsity: float) -> tuple[torch.Tensor]:
        """
        Applies fine-grained pruning to a convolutional layer.

        Parameters:
            layer (nn.Conv2d): The convolutional layer to prune.
            sparsity (float): The desired sparsity level for the layer.

        Returns:
            tuple[torch.Tensor]: A tuple containing the weight and bias masks for the pruned layer.
        """
        # Ensure sparsity_level is between 0 and 1.
        assert 0 <= sparsity <= 1, "Sparsity level must be between 0 and 1"
        
        # Gather all the weights.
        weights = []
        for param in layer.parameters():
            if param.requires_grad:
                weights.append(param.data.abs().flatten())
        
        # Get a flattened vector of all the weights.
        weights_flattened = torch.cat(weights)
        
        # Get layer magnitude threshold based on sparsity to be achieved.
        global_threshold = torch.quantile(weights_flattened,sparsity,None,interpolation="linear")
        
        # Generate mask to prune the weights and biases.
        masks = []
        for param in layer.parameters():
            if param.requires_grad:
                mask = (param.data.abs() > global_threshold)
                param.data *= mask.float()
                mask.requires_grad = False
                masks.append(mask.float())
                
        return tuple(masks)
    
    
class PrunerFilter(Pruner):
    """
    Implements filter pruning, targeting entire filters in convolutional layers.
    """
    def __init__(self,
                 module: nn.Module,
                 sparsity_levels: tuple[float]) -> None:
        """
        Initializes the filter pruner with a model/module and sparsity levels.

        Parameters:
            module (nn.Module): The model or module to apply filter pruning to.
            sparsity_levels (tuple[float]): The desired sparsity levels for each layer in the module.
        """
        super(PrunerFilter,self).__init__(module,sparsity_levels)
        
    def prune_layer(self,
                    layer: nn.Conv2d,
                    sparsity: float) -> tuple[torch.Tensor]:
        """
        Applies filter pruning to a convolutional layer.

        Parameters:
            layer (nn.Conv2d): The convolutional layer to prune.
            sparsity (float): The desired sparsity level for the layer.

        Returns:
            tuple[torch.Tensor]: A tuple containing the weight and bias masks for the pruned layer.
        """
        
        # Ensure sparsity_level is between 0 and 1.
        assert 0 <= sparsity <= 1, "Sparsity level must be between 0 and 1"
        
        # Shape of weight.
        K,C,H,W = layer.weight.shape
        
        # Convert weight of the layer into a tensor of 'channel' number of vectors.
        weight_matrix = torch.reshape(layer.weight,(K,-1))
        
        # Generate array of norms.
        frobenius_norms = torch.norm(weight_matrix, p='fro', dim=1)
        
        # Get norm magnitude threshold based on sparsity to be
        # achieved.
        norm_threshold = torch.quantile(frobenius_norms,sparsity,None,interpolation="linear")
        
        # Bias will remain unpruned.
        bias_mask = torch.ones_like(layer.bias)
        
        # Generate mask to prune the weights.
        mask_vector = (frobenius_norms>norm_threshold).float()
        expanded_mask = mask_vector[:, None, None, None]  # Expand the mask vector to add new dimensions to it. 
        weight_mask = expanded_mask.expand(-1,C,H,W)  # Add repeating elements to be same shape as weight.
        
        # Prune the layer.
        layer.weight.data *= weight_mask
        
        return (weight_mask,bias_mask)
    
    
class PrunerKernel(Pruner):
    """
    Implements kernel pruning, targeting convolutional kernels within filters.
    """
    def __init__(self,
                 module: nn.Module,
                 sparsity_levels: dict) -> None:
        """
        Initializes the kernel pruner with a model/module and sparsity levels.

        Parameters:
            module (nn.Module): The model or module to apply kernel pruning to.
            sparsity_levels (dict): The desired sparsity levels for each layer in the module.
        """
        super(PrunerKernel,self).__init__(module,sparsity_levels)    
        
    def prune_layer(self,
                    layer: nn.Conv2d,
                    sparsity: float) -> tuple[torch.Tensor]:
        """
        Applies kernel pruning to a convolutional layer.

        Parameters:
            layer (nn.Conv2d): The convolutional layer to prune.
            sparsity (float): The desired sparsity level for the layer.

        Returns:
            tuple[torch.Tensor]: A tuple containing the weight and bias masks for the pruned layer.
        """
        
        # Ensure sparsity_level is between 0 and 1.
        assert 0 <= sparsity <= 1, "Sparsity level must be between 0 and 1"
        
        # Shape of weight.
        K,C,H,W = layer.weight.shape
        
        # Generate tensor of norms. (Don't have to reshape like filter pruning)
        frobenius_norms = torch.norm(layer.weight, p='fro', dim=(2,3))
        
        # Get norm magnitude threshold based on sparsity to be
        # achieved.
        norm_threshold = torch.quantile(frobenius_norms.flatten(),sparsity,None,interpolation="linear")
        
        # Bias will remain unpruned.
        bias_mask = torch.ones_like(layer.bias)
        
        # Generate mask to prune the weights.
        mask_vector = (frobenius_norms>norm_threshold).float()
        expanded_mask = mask_vector[:, : , None, None]  # Expand the mask tensor to add new dimensions to it. 
        weight_mask = expanded_mask.expand(-1,-1,H,W)  # Add repeating elements to be same shape as weight.
        
        # Prune the layer.
        layer.weight.data *= weight_mask
        
        return (weight_mask,bias_mask)
    
        