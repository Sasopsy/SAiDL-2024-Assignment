import model
import pruner
import utils
import torch
import copy
import torch.nn as nn
from torch.utils.data import DataLoader

class SensitivityScanner():
    """
    A class for conducting sensitivity scanning on PyTorch models.

    Attributes:
        model (nn.Module): The original model to be evaluated.
        prune_module (nn.Module): The specific module within the model targeted for pruning.
        dataloader (DataLoader): The DataLoader providing the dataset for evaluation.
        device (str): The device on which computations are performed ('cpu' or 'cuda').
        criterion (torch.nn.modules.loss): The loss function used for model evaluation.
        sparsity_levels (dict): A dictionary specifying the sparsity levels for pruning, with layer names as keys.
        _pruner (str): The type of pruner to use ('finegrained', 'kernel', or 'filter').
    """
    def __init__(self,
                 _model: nn.Module,
                 dataloader: DataLoader,
                 device: str,
                 criterion: torch.nn.modules.loss,
                 _pruner: str,
                 sparsity_levels: dict) -> None:
        """
        Initializes the SensitivityScanner object with the model, pruning module, dataloader, 
        device, criterion, pruner type, sparsity levels, and masks.

        Parameters:
            _model (nn.Module): The model to be evaluated and pruned (Note: model must have attribute `features`)
            dataloader (DataLoader): The DataLoader for evaluation.
            device (str): The computation device ('cpu' or 'cuda').
            criterion (torch.nn.modules.loss): The loss function for evaluation.
            _pruner (str): The type of pruning strategy ('finegrained', 'kernel', or 'filter').
            sparsity_levels (dict): Sparsity levels for pruning, with layer names as keys.
        """
        assert _pruner == 'finegrained' or 'kernel' or 'filter', f"Pruner type {_pruner} not implemented."
        self.model = _model
        self.dataloader = dataloader
        self.device = device
        self.model.to(device)
        self.criterion = criterion
        self.sparsity_levels = sparsity_levels
        self._pruner = _pruner
        
    def evaluate_model(self,
                       _model:nn.Module):
        """
        Evaluates the model on the provided dataloader, calculating the loss and accuracy.

        Parameters:
            _model (nn.Module): The model to be evaluated.

        Returns:
            tuple: A tuple containing the loss and accuracy of the model.
        """
        
        _model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
            # Forward pass: compute the model output
            outputs = _model(inputs)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.shape[0]
            correct_predictions += (predicted == labels).sum().item()

        loss = total_loss / len(self.dataloader)
        accuracy = (correct_predictions / total_predictions)
        return loss, accuracy
    
    def fetch_pruner(self,
                     module: nn.Module,
                     sparsity_levels: dict):
        """
        Selects and initializes the appropriate pruner based on the specified pruner type.

        Parameters:
            module (nn.Module): The module to be pruned.
            sparsity_levels (dict): The sparsity levels for pruning.

        Returns:
            Pruner: An instance of the specified Pruner class.
        """
        if self._pruner == 'finegrained':
            _pruner = pruner.PrunerFinegrained(module,sparsity_levels)
        if self._pruner == 'kernel':
            _pruner = pruner.PrunerKernel(module,self.sparsity_levels)
        if self._pruner == 'filter':
            _pruner = pruner.PrunerFilter(module,self.sparsity_levels)
        return _pruner
    
    def individual_layer_scan(self,
                              key: str,
                              sparsity: float):
        """
        Evaluates the impact of pruning a single layer at a specified sparsity level.

        Parameters:
            key (str): The name of the layer to be pruned.
            sparsity (float): The sparsity level to apply to the layer.

        Returns:
            tuple: A tuple containing the loss and accuracy after pruning the layer.
        """
        dummy_model = copy.deepcopy(self.model)
        _pruner = self.fetch_pruner(dummy_model.features,self.sparsity_levels)
        _ = _pruner.prune_layer(dummy_model.features._modules[key],sparsity)
        loss,accuracy = self.evaluate_model(dummy_model)
        del dummy_model
        return loss,accuracy
    
    def layers_scan(self):
        """
        Performs a sensitivity scan across all specified layers individually to assess 
        their impact on model performance when pruned.

        Returns:
            dict: A dictionary mapping layer names to tuples of loss and accuracy.
        """
        loss_accuracy_dict = {}
        for key in self.sparsity_levels:
            loss,accuracy = self.individual_layer_scan(key,self.sparsity_levels[key])
            loss_accuracy_dict.update({key:(loss,accuracy)})

        return loss_accuracy_dict
    
    def joint_layer_scan(self):
        """
        Evaluates the model's performance after applying joint pruning to all specified layers.

        Returns:
            tuple: A tuple containing the overall loss and accuracy after joint pruning.
        """
        dummy_model = copy.deepcopy(self.model)
        _pruner = self.fetch_pruner(dummy_model.features,self.sparsity_levels)
        _ = _pruner.get_masks()
        loss,accuracy = self.evaluate_model(dummy_model)
        del dummy_model
        return loss,accuracy
    
    def custom_layer_scan(self,
                          sparsity_levels: dict):
        """
        Allows for a custom sensitivity scan with specified sparsity levels for each layer.

        Parameters:
            sparsity_levels (dict): Custom sparsity levels for pruning. 

        Returns:
            tuple: A tuple containing the loss and accuracy after applying custom pruning.
        """
        dummy_model = copy.deepcopy(self.model)
        _pruner = self.fetch_pruner(dummy_model.features,sparsity_levels)
        _ = _pruner.get_masks()
        loss,accuracy = self.evaluate_model(dummy_model)
        del dummy_model
        return loss,accuracy
        
         