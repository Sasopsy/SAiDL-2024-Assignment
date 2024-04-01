import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class TopKRouter(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 num_experts: int,
                 top_k: int) -> None:
        """
        Initializes the Naive TopKRouter with input dimensions, number of experts, and K value. 

        Parameters:
            in_channels (int): The size of the input dimension.
            num_experts (int): The total number of experts available for routing.
            top_k (int): The number of experts to be selected for each input.
        """
        super(TopKRouter,self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(in_channels,num_experts,bias=False)
        
    def forward(self,x: torch.Tensor):
        """
        Forwards the input through the router, selecting the top K experts for each input.

        Parameters:
            x(torch.Tensor): The input tensor to the model with shape (B,T,C).

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: A tuple containing the router output tensor with shape (B, T, num_experts),
                                            indices of the top K experts selected for each input, with shape (B, T, top_k).
        """
        # Shape of x: (B,T,C)
        logits = self.linear(x)
        # Shape of logits: (B,T,num_experts)
        softmax_logits = F.softmax(logits,dim=-1)
        top_k_logits, indices = softmax_logits.topk(self.top_k, dim=-1)
        # Shape of top_k_logits: (B,T,top_k)
        # Shape of indices: (B,T,top_k)
        zeros = torch.zeros_like(softmax_logits)  
        self.router_output = zeros.scatter(-1, indices, top_k_logits)
        # Shape of router_output: (B,T,num_experts)
        return self.router_output,indices
        

class TopKRouterMask(nn.Module):
    """
    Implements a Top-K Router for routing inputs to the top K experts based on their logits.

    Attributes:
        in_channels (int): The size of the input dimension.
        num_experts (int): The total number of experts available for routing.
        top_k (int): The number of experts to be selected for each input.
    """
    def __init__(self,
                 in_channels: int,
                 num_experts: int,
                 top_k: int) -> None:
        """
        Initializes the TopKRouter with input dimensions, number of experts, and K value.

        Parameters:
            in_channels (int): The size of the input dimension.
            num_experts (int): The total number of experts available for routing.
            top_k (int): The number of experts to be selected for each input.
        """
        super(TopKRouterMask,self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(in_channels,num_experts,bias=False)
        
    def forward(self,x: torch.Tensor):
        """
        Forwards the input through the router, selecting the top K experts for each input and applying masking function on them.

        Parameters:
            x(torch.Tensor): The input tensor to the model with shape (B,T,C).

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: A tuple containing the router output tensor with shape (B, T, num_experts),
                                            indices of the top K experts selected for each input, with shape (B, T, top_k).
        """
        # Shape of x: (B,T,C)
        logits = self.linear(x)
        # Shape of logits: (B,T,num_experts)
        softmax_logits = F.softmax(logits,dim=-1)
        top_k_logits, indices = softmax_logits.topk(self.top_k, dim=-1)
        # Shape of top_k_logits: (B,T,top_k)
        # Shape of indices: (B,T,top_k)
        zeros = torch.zeros_like(softmax_logits)  
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)  # Apply mask
        # Alternate gating function for sparsity.
        self.router_output = sparse_logits/sparse_logits.sum(-1,keepdim=True)
        # Shape of router_output: (B,T,num_experts)
        return self.router_output, indices
    
    

class NoisyTopKRouter(nn.Module):
    """
    Implements a Noisy Top-K Router that adds noise to the logits before routing inputs to the top K experts.

    Attributes:
        in_channels (int): The size of the input dimension.
        num_experts (int): The total number of experts available for routing.
        top_k (int): The number of experts to be selected for each input.
    """
    
    def __init__(self,
                 in_channels: int,
                 num_experts: int, 
                 top_k: int) -> None:
        """
        Initializes the NoisyTopKRouter with input dimensions, number of experts, and K value.

        Parameters:
            in_channels (int): The size of the input dimension.
            num_experts (int): The total number of experts available for routing.
            top_k (int): The number of experts to be selected for each input.
        """
        super(NoisyTopKRouter,self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(in_channels,num_experts,bias=False)
        self.noise_linear =nn.Linear(in_channels,num_experts,bias=False)    
        self.init_weights(self.topkroute_linear)
        self.init_weights(self.noise_linear)
        
    def init_weights(self,m):
        """
        Initializes the weights of a linear module with zeros.

        Parameters:
            m (nn.Module): The module to initialize.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)    
            
    def forward(self, x: torch.Tensor):
        """
        Forwards the input through the noisy router, selecting the top K experts for each input with added noise.

        Parameters:
            x (torch.Tensor): The input tensor to the model with shape (B, T, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the router output tensor with shape (B, T, num_experts),
                                             indices of the top K experts selected for each input, with shape (B, T, top_k).
        """
        self.logits = self.topkroute_linear(x)
        
        if self.training:
            self.noise_logits = self.noise_linear(x)
        
            noise = torch.randn_like(self.noise_logits)*F.softplus(self.noise_logits)
            self.noisy_logits = self.logits+noise
            
            top_k_logits, indices = self.noisy_logits.topk(self.top_k, dim=-1)
            # Shape of top_k_logits: (B,T,top_k)
            # Shape of indices: (B,T,top_k)
            zeros = torch.full_like(self.noisy_logits, float('-inf'))  

        else:
            top_k_logits, indices = self.logits.topk(self.top_k, dim=-1)
            # Shape of top_k_logits: (B,T,top_k)
            # Shape of indices: (B,T,top_k)
            zeros = torch.full_like(self.logits, float('-inf'))  
            
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)  # Change indices with top-k logits.
        # Apply softmax.
        self.router_output = F.softmax(sparse_logits, dim=-1)
        # Shape of router_output: (B,T,num_experts)
        return self.router_output, indices
    
    def kth_excluding(self,
                      tensor: torch.Tensor,
                      k: int):
        """
        Finds the k-th highest value in each row of the tensor, excluding the diagonal elements.

        Parameters:
            tensor (torch.Tensor): The input tensor.
            k (int): The rank of the element to find in each row, excluding elements on the diagonal.

        Returns:
            torch.Tensor: A tensor containing the k-th highest values for each row, excluding the diagonal same shape as tensor.
        """
        N = tensor.size(-1)
        repeated_vectors = tensor[:,None,:].expand(-1,N,N)
        mask = ~torch.eye(repeated_vectors.size(-1),dtype=torch.bool,device=repeated_vectors.device)  # Create diagonal of Falses
        mask = mask[None,:].expand_as(repeated_vectors)
        # Exclude the i'th index for every sub matrix in tensor by setting it to -inf
        exclude_ith = torch.where(mask,repeated_vectors,torch.full_like(repeated_vectors,float('-inf'))).to(repeated_vectors.device)
        topk_vals, _ = exclude_ith.topk(k,-1,sorted=True)
        kth_excluding_vals = topk_vals[:,:,k-1]
        return kth_excluding_vals
    
    def prob_in_topk(self,
                     logits: torch.Tensor,  # x.W_g 
                     noisy_logits: torch.Tensor,  # H(x)
                     noise_logits: torch.Tensor  # x.W_noise
                     ):
        """
        Calculates the probability of each logit being in the top K, considering the noise.

        Parameters:
            logits (torch.Tensor): The original logits without noise.
            noisy_logits (torch.Tensor): The logits with added noise.
            noise_logits (torch.Tensor): The logits that determine the noise level.

        Returns:
            torch.Tensor: The probabilities of each logit being in the top K.
        """
        normal_dist = Normal(0,1)
        rv = (logits-self.kth_excluding(noisy_logits,self.top_k))/F.softplus(noise_logits)
        cdf = normal_dist.cdf(rv)
        return cdf
    
    def calculate_load_balancing_loss(self):
        """
        Calculates load loss based on the variance of the probabilities of being in the top K.

        Returns:
            torch.Tensor: The load balancing loss.
        """
        # Unroll the logits.
        logits = self.logits.view(-1,self.logits.size(-1))
        noisy_logits = self.noisy_logits.view(-1,self.noisy_logits.size(-1))
        noise_logits = self.noise_logits.view(-1,self.noise_logits.size(-1))
        probs = self.prob_in_topk(logits,noisy_logits,noise_logits)
        
        load = probs.sum(0)
        eps = 1e-8
        # Calculate CV
        cv_squared_load = load.var()/(load.mean()**2+eps)
        
        return cv_squared_load
    

def get_router(router: str) -> nn.Module:
    """
    Returns the router class based on the router name provided.

    Parameters:
        router (str): The name of the router to be retrieved. Options are 'top_k' and 'noisy_top_k'.

    Returns:
        nn.Module: The router class corresponding to the provided router name.
    """
    router_dict = {'top_k': TopKRouter,
                   'top_k_mask': TopKRouterMask,
                   'noisy_top_k': NoisyTopKRouter}
    # Assert that the provided router name is in the router_dict keys
    assert router in router_dict, f"{router} is not a valid router. Choose from: {list(router_dict.keys())}"
    return router_dict[router]


class Expert(nn.Module):
    """
    Defines an expert network, which is a small feedforward neural network.

    Attributes:
        in_channels (int): The size of the input dimension.
        hidden_size (int): The size of the hidden layer within the expert.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_size: int):
        """
        Initializes the Expert with input dimensions and hidden layer size.

        Parameters:
            in_channels (int): The size of the input dimension.
            hidden_size (int): The size of the hidden layer within the expert.
        """
        super(Expert,self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(in_channels,hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,in_channels),
            nn.Dropout()
        )
        
    def forward(self,x: torch.Tensor):
        """
        Forwards the input through the expert network, producing an output for each input.

        Parameters:
            x (torch.Tensor): The input tensor to the expert.

        Returns:
            torch.Tensor: The output tensor from the expert, with the same shape as the input.
        """
        return self.expert(x)
        
    
class SparseMoE(nn.Module):
    """
    Implements a Sparse Mixture of Experts model, dynamically routing inputs to a subset of experts.

    Attributes:
        router (str): The type of router to use for directing inputs to experts. Options are 'top_k' and 'noisy_top_k'.
        in_channels (int): The size of the input dimension.
        hidden_size (int): The size of the hidden layers within the experts.
        num_experts (int): The total number of expert networks in the model.
        top_k (int): The number of experts selected for each input by the router.
    """
    
    def __init__(self,
                 router: str,
                 in_channels: int,
                 hidden_size: int,
                 num_experts: int, 
                 top_k: int) -> None:
        """
        Initializes the SparseMoE model with a specific router, input dimensions, hidden size, number of experts, and K value.

        Parameters:
            router (str): The type of router to use.
            in_channels (int): The size of the input dimension.
            hidden_size (int): The size of the hidden layers within the experts.
            num_experts (int): The total number of expert networks in the model.
            top_k (int): The number of experts selected for each input by the router.
        """
        super(SparseMoE,self).__init__()
        self.router = get_router(router)(in_channels,num_experts,top_k)
        self.experts = nn.ModuleList([Expert(in_channels,hidden_size) for _ in range(num_experts)])
        self.top_k = top_k
        
    def forward(self, x: torch.Tensor):
        """
        Forwards the input through the SparseMoE model, aggregating outputs from the top K selected experts for each input.

        Parameters:
            x (torch.Tensor): The input tensor to the model with shape (B, T, C).

        Returns:
            torch.Tensor: The final output tensor of the model, aggregating contributions from the selected experts of shape (B,T,C)
        """
        # Shape of x: (B,T,C)
        final_output = torch.zeros_like(x)
        router_output,indices = self.router(x)
        # Shape of router_output: (B,T,num_experts)
        # Shape of indices: (B,T,top_k)
        
        # Flatten batch for easier processing
        flat_x = torch.reshape(x,(-1,x.size(-1)))
        flat_router_output = router_output.view(-1, router_output.size(-1))
        # Shape of flat_x: (B*T,C) 
        # Shape of flat_router_output: (B*T,num_experts)
        
        # Loop over each expert and compute their output (if they are chosen by the router)
        for i,expert in enumerate(self.experts):
            # Create a boolean mask for the expert
            expert_mask = (indices==i).any(dim=-1)  # Boolen Tensor
            # Shape of expert_mask: (B,T), Boolen Tensor
            flat_mask = expert_mask.view(-1) 
            # Shape of flat_mask: (B*T), Boolean Tensor
            
            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                # Shape of expert_input: (expert_tokens,C)
                expert_output = expert(expert_input)        
                # Shape of expert_output: (expert_tokens,C)
                
                # Extract and apply router scores
                router_scores = flat_router_output[flat_mask,i].unsqueeze(1)
                # Shape of router_scores: (expert_tokens,1)
                weighted_output = expert_output*router_scores
                # Shape of weighted_output: (expert_tokens,C)
                
                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output
        
        return final_output