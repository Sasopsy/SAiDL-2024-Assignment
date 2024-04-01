import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from models import BaseVAE

class VAETrainer:
    """
    A trainer class for Variational Autoencoders (VAE).
    Parameters:
    - model (BaseVAE): The VAE model to train.
    - train_loader (DataLoader): DataLoader for the training data.
    - test_loader (DataLoader): DataLoader for the test data.
    - optimizer (torch.optim): Optimizer for training the model.
    - reconstruction_loss_fn (nn.Module): Reconstruction loss function for the VAE.
    - device (torch.device): The device to train the model on.
    """
    def __init__(self, 
                 model: BaseVAE, 
                 train_loader: DataLoader, 
                 test_loader: DataLoader, 
                 optimizer: torch.optim,
                 reconstruction_loss_fn: nn.Module,
                 device: torch.device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optimizer
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.history = {
            'loss_train': [],
            'loss_val': [],
            'kld_loss_train': [],
            'recon_loss_train': [],
            'kld_loss_val': [],
            'recon_loss_val': [],
        }
        
    def train_step(self,
                   data: torch.Tensor):
        """
        Performs a single training step.

        Parameters:
        - data (torch.Tensor): The input data.

        Returns:
        - float: The reconstruction loss for the step.
        - float: The KL divergence loss for the step.
        """
        data = data.to(self.device)
        self.optimizer.zero_grad()
        reconstruction, parameter1, parameter2 = self.model(data)
        recon_loss = self.reconstruction_loss_fn(reconstruction,data)
        kld_loss = self.model.compute_kl_divergence(parameter1, parameter2)
        loss = recon_loss + kld_loss
        loss.backward()
        self.optimizer.step()
        return recon_loss.item(), kld_loss.item()
    
    def eval_step(self, 
                  data: torch.Tensor):
        """
        Performs a single evaluation step on a batch of data.

        Parameters:
        - data (torch.Tensor): The input data to evaluate.

        Returns:
        - float: The average reconstruction loss over the batch.
        - float: The average KL divergence loss over the batch.
        """
        data = data.to(self.device)
        reconstruction, parameter1, parameter2 = self.model(data)
        recon_loss = self.reconstruction_loss_fn(reconstruction,data)
        kld = self.model.compute_kl_divergence(parameter1, parameter2)
        return recon_loss.item(), kld.item()

    def train(self, 
              epochs: int):
        """
        Trains the VAE model for a specified number of epochs.

        Parameters:
        - epochs (int): Number of epochs to train the model.
        """
        for epoch in range(epochs):
            overall_loss = 0
            overall_kld_loss = 0
            overall_recon_loss = 0
            
            # Training
            train_loop = tqdm(self.train_loader, unit="batch", desc=f"Epoch {epoch+1}/{epochs}")
            self.model.train()
            for data, _ in train_loop:
                recon_loss, kld_loss = self.train_step(data)
                loss = recon_loss + kld_loss
                overall_loss += loss
                overall_kld_loss += kld_loss
                overall_recon_loss += recon_loss
                    
                # Postfix for the tqdm bar
                train_loop.set_postfix(loss=loss,
                                        kld_loss=kld_loss,
                                        recon_loss=recon_loss)
            
            # Save training history
            avg_loss_train = overall_loss / len(self.train_loader.dataset)
            avg_kld_loss_train = overall_kld_loss / len(self.train_loader.dataset)
            avg_recon_loss_train = overall_recon_loss / len(self.train_loader.dataset)
            
            self.history['loss_train'].append(avg_loss_train)
            self.history['kld_loss_train'].append(avg_kld_loss_train)
            self.history['recon_loss_train'].append(avg_recon_loss_train)

            # Print training summary for the epoch
            print(f"Epoch {epoch + 1} Training - Total Loss: {avg_loss_train:.4f}, KLD Loss: {avg_kld_loss_train:.4f}, Recon Loss: {avg_recon_loss_train:.4f}")

            # Evaluation
            self.evaluate(epoch)
            
    def evaluate(self, epoch: int):
        """
        Evaluates the VAE model on the test dataset after each training epoch.

        Parameters:
        - epoch (int): Current epoch number.
        """
        total_recon_loss = 0
        total_kld = 0
        with torch.no_grad():
            test_loop=tqdm(self.test_loader, unit="batch", desc="Evaluating")
            self.model.eval()
            for data, _ in test_loop:
                recon_loss, kld = self.eval_step(data)
                total_recon_loss += recon_loss
                total_kld += kld

                # Postfix for the tqdm bar
                test_loop.set_postfix(recon_loss=recon_loss,
                                      kld_loss=kld)
        
        # Save evaluation history
        avg_recon_loss_val = total_recon_loss / len(self.test_loader.dataset)
        avg_kld_loss_val = total_kld / len(self.test_loader.dataset)
        avg_loss_val = avg_recon_loss_val + avg_kld_loss_val

        self.history['loss_val'].append(avg_loss_val)
        self.history['kld_loss_val'].append(avg_kld_loss_val)
        self.history['recon_loss_val'].append(avg_recon_loss_val)

        # Print evaluation summary for the epoch
        print(f"Epoch {epoch + 1} Evaluation - Total Loss: {avg_loss_val:.4f}, KLD Loss: {avg_kld_loss_val:.4f}, Recon Loss: {avg_recon_loss_val:.4f}")
        
