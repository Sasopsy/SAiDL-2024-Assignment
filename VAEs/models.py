import torch
import torch.nn as nn
from torch.distributions import Normal,Beta,Gamma

class Reshape(nn.Module):
    """
    A module that reshapes its input to the specified output shape.

    Parameters:
    - out_shape (tuple[int]): The desired output shape.
    """
    def __init__(self,out_shape: tuple[int]):
        super(Reshape,self).__init__()
        self.reshape = torch.reshape
        self.out_shape = out_shape
        
    def forward(self,x):
        """
        Reshapes the input tensor to the specified output shape.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The reshaped tensor.
        """
        return self.reshape(x,shape=(x.shape[0],)+self.out_shape)


class BaseVAE(nn.Module):
    """
    Base class for Variational Autoencoders (VAE).

    Parameters:
    - z_dim (int): The dimensionality of the latent space.
    """
    def __init__(self,
                 z_dim: int):
        super(BaseVAE,self).__init__()
        self.z_dim = z_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # MNIST images are grayscale, hence input channels is 1
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),  
            nn.Flatten())
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim,64),
            Reshape((64,1,1)),
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid())
        
    def encode(self,x: torch.Tensor):
        """
        Encodes the input into a latent representation. Must be implemented by subclasses.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - The latent representation of the input.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reparameterise(self,
                       param1: torch.Tensor,
                       param2: torch.Tensor):
        """
        Implements the reparameterization trick for Variational Autoencoders (VAEs).

        Parameters:
        - param1 (torch.Tensor): First parameter of distribution.
        - param2 (torch.Tensor): Second parameter of distribution.
        
        Returns:
        - torch.Tensor: A sampled tensor from the desired distribution.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    
    def decode(self,z: torch.Tensor):
        """
        Decodes the latent representation back into the input space. Must be implemented by subclasses.

        Parameters:
        - z (torch.Tensor): The latent representation.

        Returns:
        - The reconstructed input.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def forward(self,x: torch.Tensor):
        """
        Defines the forward pass of the VAE. Must be implemented by subclasses.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - The output of the VAE forward pass.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def compute_kl_divergence(self, 
                              parameter1: torch.Tensor,
                              parameter2: torch.Tensor,):
        """
        Computes the KL divergence between the latent distribution and the prior.

        Parameters:
        - parameter1 (torch.Tensor): Parameter 1 for distribution.
        - parameter2 (torch.Tensor): Parameter 2 for distribution.

        Returns:
        torch.Tensor: KL divergence.
        """
        
    def sample(self,
               num_samples: int,
               device: torch.device):
        """
        Samples from the prior distribution and decodes it to generate new data.

        Parameters:
        - num_samples (int): The number of samples to generate.

        Returns:
        - torch.Tensor: The generated data of shape (num_samples,1,28,28)
        """
        with torch.no_grad():
            z = self.prior_dist.rsample((num_samples,self.z_dim)).to(device)
            generated_images = self.decode(z)
        return generated_images
 
    
class NormalVAE(BaseVAE):
    """
    Implements a Variational Autoencoder with a Normal distribution in the latent space.

    Parameters:
    - z_dim (int): The dimensionality of the latent space.
    - prior_mean (float): The mean of the prior Normal distribution.
    - prior_std (float): The standard deviation of the prior Normal distribution.
    """
    def __init__(self,
                 z_dim: int,
                 prior_mean: float,
                 prior_std: float):
        """
        Initializes the NormalVAE model with specified dimensions and prior parameters.
        """
        super(NormalVAE,self).__init__(z_dim)
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.prior_dist = Normal(prior_mean,prior_std)
        self.z_dim=z_dim
        
        self.mean_layer = nn.Linear(64,z_dim)
        self.log_var_layer = nn.Linear(64,z_dim)
        
    def encode(self,
               x: torch.Tensor):
        """
        Encodes input images into a distribution over the latent space.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mean and log variance of the latent distribution.
        """
        x = self.encoder(x)
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        return mean,log_var
    
    def reparameterise(self,
                       mean: torch.Tensor, 
                       log_var: torch.Tensor):
        """
        Performs reparameterization by sampling from a Gaussian distribution parameterized by the given mean and log variance.

        Parameters:
        - mean (torch.Tensor): The mean vector of the Gaussian distribution for each element in the latent space.
        - log_var (torch.Tensor): The log variance vector of the Gaussian distribution for each element in the latent space. 

        Returns:
        - torch.Tensor: A tensor of sampled values from the Gaussian distribution parameterized by `mean` and `log_var`.
        """
        dist = Normal(mean,torch.exp(log_var*0.5))
        z = dist.rsample()
        return z
        
    def decode(self,z: torch.Tensor):
        """
        Decodes latent vectors back into reconstructed images.

        Parameters:
        - z (torch.Tensor): Latent vector.

        Returns:
        torch.Tensor: Reconstructed images.
        """
        x_reconstructed = self.decoder(z)
        return x_reconstructed
    
    def forward(self,x: torch.Tensor):
        """
        Performs a forward pass of the model, encoding an input, sampling from the latent space, and decoding to reconstruct the input.

        Parameters:
        - x (torch.Tensor): The input images.

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed images, mean, and log variance of the latent distribution.
        """
        mean,log_var = self.encode(x)
        z = self.reparameterise(mean,log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed,mean,log_var
    
    def compute_kl_divergence(self, 
                              mean: torch.Tensor,
                              log_var: torch.Tensor,):
        """
        Computes the men KL divergence between the latent distribution and the prior.

        Parameters:
        - mean (torch.Tensor): Mean of the latent distribution.
        - log_var (torch.Tensor): Log variance of the latent distribution.

        Returns:
        torch.Tensor: Average KL divergence of the observations.
        """
        KLD = torch.distributions.kl_divergence(
            Normal(mean,torch.exp(log_var*0.5)),
            self.prior_dist
        )
        return KLD.sum()

    
class GammaVAE(BaseVAE):
    """
    Implements a Variational Autoencoder with a Gamma distribution in the latent space.

    Parameters:
    - z_dim (int): The dimensionality of the latent space.
    - prior_k (float): The shape parameter of the prior Gamma distribution.
    - prior_theta (float): The scale parameter of the prior Gamma distribution.
    """
    def __init__(self, 
                 z_dim: int, 
                 prior_k: float, 
                 prior_theta: float):
        """
        Initializes the GammaVAE model with specified dimensions and prior parameters.
        """
        super(GammaVAE, self).__init__(z_dim)
        self.prior_alpha = prior_k
        self.prior_beta = prior_theta
        self.prior_dist = Gamma(prior_k, prior_theta)
        
        self.log_k_layer = nn.Linear(64, z_dim)  # For shape parameter
        self.log_theta_layer = nn.Linear(64, z_dim)   # For rate parameter
    
    def encode(self, x: torch.Tensor):
        """
        Encodes input images into parameters of a Gamma distribution over the latent space.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Log k and log theta of the latent Gamma distribution.
        """
        x = self.encoder(x)
        log_k = self.log_k_layer(x)
        log_theta = self.log_theta_layer(x)
        return log_k, log_theta
    
    def reparameterise(self,
                       log_k: torch.Tensor,
                       log_theta: torch.Tensor):
        dist = Gamma(torch.exp(log_k), torch.exp(log_theta))
        z = dist.rsample()  # Use rsample for reparameterization
        return z

    
    def decode(self, z: torch.Tensor):
        """
        Decodes latent vectors back into reconstructed images.

        Parameters:
        - z (torch.Tensor): Latent vector.

        Returns:
        torch.Tensor: Reconstructed images.
        """
        x_reconstructed = self.decoder(z)
        return x_reconstructed
    
    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass of the model, encoding an input, sampling from the latent Gamma distribution, and decoding to reconstruct the input.

        Parameters:
        - x (torch.Tensor): The input images.

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed images, log k, and log theta of the latent distribution.
        """
        log_k, log_theta = self.encode(x)
        z = self.reparameterise(log_k,log_theta)
        x_reconstructed = self.decode(z)
        return x_reconstructed, log_k, log_theta
    
    def compute_kl_divergence(self,
                              log_k: torch.Tensor,
                              log_theta: torch.Tensor):
        """
        Computes the KL divergence between the latent Gamma distribution and the prior.

        Parameters:
        - log_k (torch.Tensor): Log shape parameter of the latent Gamma distribution.
        - log_theta (torch.Tensor): Log rate parameter of the latent Gamma distribution.

        Returns:
        torch.Tensor: KL divergence.
        """
        KLD = torch.distributions.kl_divergence(
            Gamma(torch.exp(log_k), torch.exp(log_theta)),
            self.prior_dist
        )
        # Return negative kld for final loss function.
        return KLD.sum()
    
    
class ClassifierMNIST(nn.Module):
    """
    A medium-sized CNN model for classifying MNIST digits.
    """
    def __init__(self):
        super(ClassifierMNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, stride=1 , kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, stride=2, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, stride=2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10),
        )
    
    def forward(self,
                x: torch.Tensor):
        """
        Defines the forward pass of the classifier.

        Parameters:
        - x (torch.Tensor): The input image tensor.

        Returns:
        - torch.Tensor: The output logits of the classifier.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        