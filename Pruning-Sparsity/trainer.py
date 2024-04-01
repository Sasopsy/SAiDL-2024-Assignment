import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Tuple
from tqdm.auto import tqdm
import utils
from dataclasses import dataclass

@dataclass
class TrainConfigs:
    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 64  # Adjusted for more realistic training
    loss_fn: torch.nn.modules.loss = nn.CrossEntropyLoss()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    def __init__(self,
                 configs: TrainConfigs,
                 model: torch.nn.Module,
                 optim: Optimizer,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 prune_masks: dict = None,
                 ) -> None:
        """
        Initializes the Trainer object with training configurations, model, optimizer, data loaders, and optional pruning masks.

        Args:
            configs (TrainConfigs): Configuration object containing training parameters.
            model (torch.nn.Module): The model to train.
            optim (Optimizer): Optimizer used for training.
            train_dataloader (DataLoader): DataLoader for training data.
            val_dataloader (DataLoader): DataLoader for validation data.
            prune_masks (dict, optional): Dictionary of pruning masks for model parameters.
        """
        self.configs = configs
        self.model = model
        self.optim = optim
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.prune_masks = prune_masks
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        self.device = self.configs.device

    def train_one_epoch(self) -> Tuple[float, float]:
        """
        Trains the model for one epoch through the training dataset.

        Returns:
            Tuple[float, float]: The average loss and accuracy for the epoch.
        """
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        progress_bar = tqdm(self.train_dataloader, desc='Training', leave=False)
        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            self.optim.zero_grad()
            output = self.model(data)
            loss = self.configs.loss_fn(output, target)
            loss.backward()
            
            if self.prune_masks is not None:
                utils.apply_pruning_mask(self.model.features,self.prune_masks)
            
            self.optim.step()

            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += data.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss/total_samples:.4f}',
                'accuracy': f'{total_correct/total_samples:.4f}'
            })

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """
        Validates the model on the validation dataset.

        Returns:
            Tuple[float, float]: The average loss and accuracy on the validation dataset.
        """
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for data, target in tqdm(self.val_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.configs.loss_fn(output, target)
                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == target).sum().item()
                total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def train(self):
        """
        Executes the training process over a specified number of epochs, including both training and validation phases.
        Stores training history for loss and accuracy.
        """
        for epoch in range(self.configs.epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f'Epoch {epoch+1}/{self.configs.epochs}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    def save_model(self, path: str):
        """
        Saves the model's state dictionary to a specified file path and also prune masks.

        Args:
            path (str): The file path where the model should be saved.
        """
        torch.save({
            'model_state_dict':self.model.state_dict(),
            'prune_masks':self.prune_masks
            }, path)


    def load_model(self, path: str):
        """
        Loads the model's state dictionary from a specified file path.

        Args:
            path (str): The file path from where the model should be loaded.
        """
        dictionary = torch.load(path)
        self.model.load_state_dict(dictionary['model_state_dict'])

