import torch
from torch import nn
from torch.optim import *
from torchvision.datasets import *
from torchvision.transforms import *
from torch.optim.lr_scheduler import *


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def count_layer_instances(module: nn.Module, 
                          layer_type: type):
    """
    Counts the instances of a specific layer type in a PyTorch model.

    Parameters:
        module (nn.Module): The module to search through.
        layer_type (type): The type of layer to count (e.g., nn.Conv2d).

    Returns:
        int: The count of instances of the specified layer type.
    """
    count = 0
    for layer in module:
        if isinstance(layer, layer_type):
            count += 1
    return count


def apply_pruning_mask(module: nn.Module,
                       masks: dict):
    """Applies pruning mask to the gradients. Call after backward propagation and before optimizer step.

    Args:
        module (nn.Module): Module or model to be applied mask to.
        masks (dict): Pruning masks.
    """
    with torch.no_grad():
        for keys in masks:
            weight_mask = masks[keys][0]
            bias_mask = masks[keys][1]
            # Apply mask.
            module._modules[keys].weight.grad *= weight_mask
            module._modules[keys].bias.grad *= bias_mask
        
