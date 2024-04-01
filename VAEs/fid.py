import torch
from torch import nn
from torchvision.models import inception_v3,Inception_V3_Weights
import torch.nn.functional as F
from models import BaseVAE
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from scipy import linalg
import numpy as np
import torchvision.transforms as transforms


class PartialInceptionNetwork(nn.Module):
    """
    A partial Inception network module used for obtaining the features from an intermediate layer.

    Attributes:
    - inception_network (torch.nn.Module): The Inception v3 network pre-loaded with weights.
    - transform_input (bool): Whether to transform the input images to the expected Inception v3 format.
    """
    def __init__(self, transform_input=True):
        """ 
        Initializes the PartialInceptionNetwork module.

        Args:
        - transform_input (bool): If True, transforms input images to the expected Inception v3 format.
        """
        super().__init__()
        self.inception_network = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input
        self.inception_network.eval()

    def output_hook(self, module, input, output):
        """
        A hook to be attached to an Inception network layer to grab the output of that layer.

        Args:
        - module (torch.nn.Module): The module the hook is attached to.
        - input (torch.Tensor): The input to the module.
        - output (torch.Tensor): The output from the module.
        """
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Forward pass through the partial Inception network.

        Args:
        - x (torch.Tensor): Input tensor of shape (N, 3, 299, 299).

        Returns:
        - torch.Tensor: Activations from the Mixed_7c layer, reshaped to (N, 2048).
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = F.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations
    
    
class FID:
    """
    FID (Fréchet Inception Distance) calculation module for evaluating the quality of generated images.

    Attributes:
    - device (torch.device): The device to perform calculations on.
    - model (BaseVAE): The generative model from which images are generated.
    - val_dataset (Dataset): The validation dataset to compare against the generated images.
    - batch_size (int): The batch size for processing images.
    """
    def __init__(self,
                 device: torch.device,
                 model: BaseVAE,
                 val_dataset: Dataset,
                 batch_size: int):
        """
        Initializes the FID calculation module.

        Args:
        - device (torch.device): The device to perform calculations on.
        - model (BaseVAE): The VAE model.
        - val_dataset (Dataset): The validation dataset to compare against the generated images.
        - batch_size (int): The batch size for processing images separately.
        """
        assert len(val_dataset)%batch_size == 0, "Ensure batch size is divisible by length of val dataset."
        self.val_dataset = val_dataset
        self.val_dataloader = DataLoader(val_dataset,batch_size)
        self.device = device
        self.inception_model = PartialInceptionNetwork().to(device)
        self.inception_model.eval()
        self.model = model
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
        ])
        
    def _generate_images(self):
        """
        Generates images using the model and stores them in self.generated_images.
        """
        self.generated_images = []
        print("Generating images ...")
        
        loop = tqdm(range(len(self.val_dataset)//self.batch_size),unit="batches")
        for i in loop:
            images = self.model.sample(self.batch_size,self.device)
            self.generated_images.append(images)
        
    def _calculate_features_real(self):
        """
        Calculates the features of real images using the Inception model and creates a numpy matrix of it.
        """
        features_list_real = []
        loop = tqdm(self.val_dataloader,unit="batches")
        print("Calculating features of real images ...")
        with torch.no_grad():
            for data, _ in loop:
                if len(data.shape) == 3:  # Add channel if it's missing
                    data = data.unsqueeze(1)
                data = data.repeat(1, 3, 1, 1)  # Repeat channel to simulate RGB
                data = self.transform(data)
                features = self.inception_model(data.to(self.device))  # (N,2048)
                features_list_real.append(features)
        self.features_real = torch.concat(features_list_real,dim=0).cpu().detach().numpy()
    
    def _calculate_features_fake(self):
        """
        Calculates the features of generated (fake) images using the Inception model and creates a numpy matrix of it.
        """
        features_list_fake = []
        loop = tqdm(self.generated_images,unit="batches")
        print("Calculating features of fake images ...")
        with torch.no_grad():
            for data in loop:
                if len(data.shape) == 3:  # Add channel if it's missing
                    data = data.unsqueeze(1)
                data = data.repeat(1, 3, 1, 1)  # Repeat channel to simulate RGB
                data = self.transform(data)
                features = self.inception_model(data.to(self.device))  # (N,2048)
                features_list_fake.append(features)
        self.features_fake = torch.concat(features_list_fake,dim=0).cpu().detach().numpy()
        
    def calculate_fid(self):
        """
        Calculates the Fréchet Inception Distance (FID) between generated and real images.

        Returns:
        - float: The calculated FID score.
        """
        self._generate_images()
        self._calculate_features_fake()
        self._calculate_features_real()
    
        mu_real, sigma_real = self.features_real.mean(axis=0), np.cov(self.features_real,rowvar=False)
        mu_fake, sigma_fake = self.features_fake.mean(axis=0), np.cov(self.features_fake,rowvar=False)
                
        mu_diff = mu_real - mu_fake
                
        mean_dist = np.dot(mu_diff,mu_diff)
        covmean, _ = linalg.sqrtm(sigma_fake.dot(sigma_real), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = mean_dist + np.trace(sigma_real + sigma_fake - 2.0*covmean)

        return fid