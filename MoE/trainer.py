import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import ConllLSTM_MoE
from models import SquadLSTM_MoE
from moe import NoisyTopKRouter

class ConllTrainer:
    """
    Trainer class for a CoNLL dataset using an LSTM model with Mixture of Experts (MoE).

    Attributes:
        model (ConllLSTM_MoE): The LSTM model with MoE layer.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim): Optimizer for the model.
        device (torch.device): Device on which the model will run.
        importance_cv_squared (bool): Flag to add coefficient of variation squared term for router importance to the loss.
        w_importance (float): Weight of the importance coefficient of variation squared term in the loss.
        load_cv_squared (bool): Flag to add load balancing loss for NoisyTopKRouter.
        w_load (float): Weight of the load balancing loss in the total loss.
    """
    def __init__(self, 
                 model: ConllLSTM_MoE, 
                 train_dataloader: DataLoader, 
                 val_dataloader: DataLoader, 
                 optimizer: torch.optim, 
                 device: torch.device,
                 importance_cv_squared: bool = False,
                 w_importance: float = 0,
                 load_cv_squared: bool = False,
                 w_load: float = 0):
        """
        Initializes the trainer with model, dataloaders, optimizer, device, and optional loss components.
        """
        self.model = model.to(device)
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.optimizer = optimizer
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.device = device
        self.importance_cv_squared = importance_cv_squared
        self.w_importance = w_importance
        self.load_cv_squared = load_cv_squared
        self.w_load = w_load
        if load_cv_squared:
            assert isinstance(self.model.moe.router,NoisyTopKRouter), "Load balancing only on noisy top-k router."
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def calculate_cv_squared_importance(self,
                             router_output: torch.Tensor):
        """
        Calculates the coefficient of variation squared for router outputs to evaluate importance across batches.

        Parameters:
            router_output (torch.Tensor): Output tensor from the router indicating expert assignments.

        Returns:
            float: Coefficient of variation squared of the router's importance scores.
        """
        importance = router_output.float().sum(dim=0)
        # Avoid division by zero by adding a small epsilon where mean is zero
        eps = 1e-8
        cv_sqaured = importance.var()/(importance.mean()**2+eps)
        return cv_sqaured

    def train_step(self):
        """
        Performs a single training step over the training dataset.

        Returns:
            Tuple[float, float]: Average loss and accuracy over the training set for the step.
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        train_loop = tqdm(self.train_loader, unit='batch')
        for inputs,labels in train_loop:
            # Move batch to device
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            predictions = outputs.argmax(dim=2)

            # Generate mask for non-padded labels
            mask = (labels != -100)
            loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # Get moe router output and calculate importance cv-squared
            if self.importance_cv_squared:
                loss += self.w_importance*self.calculate_cv_squared_importance(self.model.moe.router.router_output)
                 
            if self.load_cv_squared:
                loss += self.w_load*self.model.moe.router.calculate_load_balancing_loss()
                
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            total_loss += loss.item()
            correct_predictions += (predictions[mask] == labels[mask]).sum().item()
            total_predictions += mask.sum().item()
            
            # Add postfix
            train_loop.set_postfix(loss = loss.item())

        # Record loss and accuracy
        average_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        return average_loss, accuracy

    def validate(self):
        """
        Validates the model on the validation dataset.

        Returns:
            Tuple[float, float]: Average loss and accuracy over the validation set.
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            val_loop = tqdm(self.val_loader, unit='batch')
            for inputs,labels in val_loop:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                predictions = outputs.argmax(dim=2)
                mask = (labels != -100)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_loss += loss.item()
                correct_predictions += (predictions[mask] == labels[mask]).sum().item()
                total_predictions += mask.sum().item()
                
                # Add postfix
                val_loop.set_postfix(loss = loss.item())

        average_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
        return average_loss, accuracy

    def train(self, epochs):
        """
        Trains the model for a specified number of epochs, evaluating against the validation set after each epoch.

        Parameters:
            epochs (int): Number of epochs to train the model.
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.validate()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}")
            print(f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.4f}")
            print(" ")
            

class SquadTrainer:
    """
    Trainer class for the SQuAD dataset using an LSTM model with Mixture of Experts (MoE).

    Attributes:
        model (SquadLSTM_MoE): The LSTM model with MoE layer.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim): Optimizer for the model.
        device (torch.device): Device on which the model will run.
        importance_cv_squared (bool): Flag to add coefficient of variation squared term for router importance to the loss.
        w_importance (float): Weight of the importance coefficient of variation squared term in the loss.
        load_cv_squared (bool): Flag to add load balancing loss for NoisyTopKRouter.
        w_load (float): Weight of the load balancing loss in the total loss.
    """
    def __init__(self, 
                 model: SquadLSTM_MoE, 
                 train_dataloader: DataLoader, 
                 val_dataloader: DataLoader, 
                 optimizer: torch.optim, 
                 device: torch.device,
                 importance_cv_squared: bool = False,
                 w_importance: float = 0,
                 load_cv_squared: bool = False,
                 w_load: float = 0):
        """
        Initializes the trainer with model, dataloaders, optimizer, device, and optional loss components.
        """
        self.model = model.to(device)
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.optimizer = optimizer
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.device = device
        self.importance_cv_squared = importance_cv_squared
        self.w_importance = w_importance
        self.load_cv_squared = load_cv_squared
        self.w_load = w_load
        if load_cv_squared:
            assert isinstance(self.model.moe.router,NoisyTopKRouter), "Load balancing only on noisy top-k router."
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def calculate_cv_squared_importance(self,
                             router_output: torch.Tensor):
        """
        Calculates the coefficient of variation squared for router outputs to evaluate importance across batches.

        Parameters:
            router_output (torch.Tensor): Output tensor from the router indicating expert assignments.

        Returns:
            float: Coefficient of variation squared of the router's importance scores.
        """
        importance = router_output.float().sum(dim=0)
        # Avoid division by zero by adding a small epsilon where mean is zero
        eps = 1e-8
        cv_sqaured = importance.var()/(importance.mean()**2+eps)
        return cv_sqaured
        
    def train_step(self):
        """
        Performs a single training step over the training dataset.

        Returns:
            Tuple[float, float]: Average loss and accuracy over the training set for the step.
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        train_loop = tqdm(self.train_loader, unit='batch')
        for batch in train_loop:
            # Move batch to device
            context, question,ans_start,ans_end = [b.to(self.device) for b in batch]

            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(context,question)
            start_logits, end_logits = outputs
            start_predictions = start_logits.argmax(1)
            end_predictions = end_logits.argmax(1)

            # Loss 
            start_loss = self.loss_fn(start_logits,ans_start)
            end_loss = self.loss_fn(end_logits,ans_end)
            loss = start_loss + end_loss
            
            # Get moe router output and calculate importance cv-squared
            if self.importance_cv_squared:
                loss += self.w_importance*self.calculate_cv_squared_importance(self.model.moe.router.router_output)
                 
            if self.load_cv_squared:
                loss += self.w_load*self.model.moe.router.calculate_load_balancing_loss()
                
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            total_loss += loss.item()
            correct_predictions += (start_predictions==ans_start).sum().item() + (end_predictions==ans_end).sum().item() 
            total_predictions += ans_start.size(0) + ans_end.size(0)
            
            # Add postfix
            train_loop.set_postfix(loss = loss.item())

        # Record loss and accuracy
        average_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        return average_loss, accuracy
    
    def validate(self):
        """
        Validates the model on the validation dataset.

        Returns:
            Tuple[float, float]: Average loss and accuracy over the validation set.
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            val_loop = tqdm(self.val_loader, unit='batch')
            for batch in val_loop:
                context, question,ans_start,ans_end = [b.to(self.device) for b in batch]
                
                outputs = self.model(context,question)
                start_logits, end_logits = outputs
                start_predictions = start_logits.argmax(1)
                end_predictions = end_logits.argmax(1)
                
                start_loss = self.loss_fn(start_logits,ans_start)
                end_loss = self.loss_fn(end_logits,ans_end)
                loss = start_loss + end_loss
                total_loss += loss.item()
                
                correct_predictions += (start_predictions==ans_start).sum().item() + (end_predictions==ans_end).sum().item() 
                total_predictions += start_predictions.size(0)
                
                # Add postfix
                val_loop.set_postfix(loss = loss.item())

        average_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
        return average_loss, accuracy
    
    def train(self, epochs):
        """
        Trains the model for a specified number of epochs, evaluating against the validation set after each epoch.

        Parameters:
            epochs (int): Number of epochs to train the model.
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.validate()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}")
            print(f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.4f}")
            print(" ")