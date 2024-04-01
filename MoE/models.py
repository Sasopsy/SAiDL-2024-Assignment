import torch
import torch.nn as nn
from moe import SparseMoE
from moe import Expert


class ConnllBaseLSTM(nn.Module):
    """
    A basic LSTM model enhanced with an expert layer for sequence modeling tasks.

    Attributes:
        embedding (nn.Embedding): Embedding layer for token vector representations.
        lstm1 (nn.LSTM): The first LSTM layer for initial sequence processing.
        expert (Expert): A custom expert layer for intermediate processing.
        lstm2 (nn.LSTM): The second LSTM layer for further sequence processing.
        fc (nn.Linear): Final classification layer for output predictions.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 intermediate_expert_dim: int,
                 output_dim: int,
                 model_state_dict: dict,
                 num_layers: int = 1,
                 ) -> None:
        """
        Initializes the BaseLSTM model with specified parameters.

        Parameters:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of token embeddings.
            hidden_dim (int): Dimensionality of LSTM hidden states.
            intermediate_expert_dim (int): Dimensionality of the expert layer's hidden layer.
            output_dim (int): Dimensionality of the final output predictions.
            model_state_dict (dict): State dictionary to initialize the embedding layer.
            num_layers (int): Number of layers in each LSTM. Defaults to 1.
        """
        super(ConnllBaseLSTM,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.embedding.load_state_dict(model_state_dict)
        for params in self.embedding.parameters():
            params.requires_grad = False
        self.lstm1 = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True,batch_first=True,num_layers=num_layers)
        
        self.expert = Expert(hidden_dim*2,intermediate_expert_dim)
        
        self.lstm2 = nn.LSTM(hidden_dim*2,hidden_dim,bidirectional=True,batch_first=True,num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim*2,output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BaseLSTM model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, sequence_length, output_dim).
        """
        # Shape of x: (B,T)
        embedded = self.embedding(x)
        # Shape of embedded: (B,T,embedding_dim)
        lstm1_out, _ = self.lstm1(embedded)
        # Shape of lstm1_out: (B,T,2*hidden_dim)
        expert_output = self.expert(lstm1_out)
        # Shape of expert_output: (B,T,2*hidden_dim)
        lstm2_out, _ = self.lstm2(expert_output)
        # Shape of of lstm2_out: (B,T,2*hidden_dim)
        predictions = self.fc(lstm2_out)
        # Shape of predictions: (B,T,out_dim)
        return predictions
    

class ConllLSTM_MoE(nn.Module):
    """
    An LSTM model with a Mixture of Experts (MoE) layer for enhanced sequence modeling.

    Attributes:
        embedding (nn.Embedding): Embedding layer for token vector representations.
        lstm1 (nn.LSTM): The first LSTM layer for initial sequence processing.
        moe (SparseMoE): Sparse Mixture of Experts layer for selective expert processing.
        lstm2 (nn.LSTM): The second LSTM layer for further sequence processing post-expert routing.
        fc (nn.Linear): Final classification layer for output predictions.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 intermediate_expert_dim: int,
                 output_dim: int,
                 model_state_dict: dict,
                 router: str,
                 num_experts: int,
                 top_k: int,
                 num_layers: int = 1,
                 ) -> None:
        """
        Initializes the LSTM_MoE model with specified parameters and MoE configuration.

        Parameters:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of token embeddings.
            hidden_dim (int): Dimensionality of LSTM hidden states.
            intermediate_expert_dim (int): Dimensionality of the expert layer's hiddem layer.
            output_dim (int): Dimensionality of the final output predictions.
            model_state_dict (dict): State dictionary to initialize the embedding layer.
            router (str): Type of router used in the MoE layer.
            num_experts (int): Number of experts in the MoE layer.
            num_layers (int): Number of layers in each LSTM. Defaults to 1.
        """
        super(ConllLSTM_MoE,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        for params in self.embedding.parameters():
            params.requires_grad = False
        self.embedding.load_state_dict(model_state_dict)
        self.lstm1 = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True,batch_first=True,num_layers=num_layers)
        
        self.moe = SparseMoE(router,2*hidden_dim,intermediate_expert_dim,num_experts,top_k=top_k)
        
        self.lstm2 = nn.LSTM(hidden_dim*2,hidden_dim,bidirectional=True,batch_first=True,num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim*2,output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM_MoE model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, sequence_length, output_dim).
        """
        # Shape of x: (B,T)
        embedded = self.embedding(x)
        # Shape of embedded: (B,T,embedding_dim)
        lstm1_out, _ = self.lstm1(embedded)
        # Shape of lstm1_out: (B,T,2*hidden_dim)
        expert_output = self.moe(lstm1_out)
        # Shape of expert_output: (B,T,2*hidden_dim)
        lstm2_out, _ = self.lstm2(expert_output)
        # Shape of of lstm2_out: (B,T,2*hidden_dim)
        predictions = self.fc(lstm2_out)
        # Shape of predictions: (B,T,out_dim)
        return predictions
    

class SquadBaseLSTM(nn.Module):
    """
    A basic LSTM model for the SQuAD question answering task, incorporating an expert layer for processing.

    Attributes:
        embedding (nn.Embedding): Embedding layer for token vector representations.
        lstm1 (nn.LSTM): The first LSTM layer for initial processing of combined context and question inputs.
        expert (Expert): A custom expert layer for specialized processing of LSTM outputs.
        lstm2 (nn.LSTM): The second LSTM layer for further processing after the expert layer.
        output_linear (nn.Linear): Linear layer for predicting start and end indices of answers in the context.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 intermediate_expert_dim: int,
                 model_state_dict: dict,
                 num_layers: int = 1,):
        """
        Initializes the SquadBaseLSTM model with specified parameters.

        Parameters:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of token embeddings.
            hidden_dim (int): Dimensionality of LSTM hidden states.
            intermediate_expert_dim (int): Dimensionality of the expert layer's hidden layer.
            model_state_dict (dict): State dictionary to initialize the embedding layer.
            num_layers (int): Number of layers in each LSTM. Defaults to 1.
        """
        super(SquadBaseLSTM,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.embedding.load_state_dict(model_state_dict)
        for params in self.embedding.parameters():
            params.requires_grad = False
        self.lstm1 = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True,batch_first=True,num_layers=num_layers)
        
        self.expert = Expert(hidden_dim*2,intermediate_expert_dim)
        
        self.lstm2 = nn.LSTM(hidden_dim*2,hidden_dim,bidirectional=True,batch_first=True,num_layers=num_layers)
        
        self.output_linear = nn.Linear(hidden_dim*2,1)
        
    def forward(self,context,question):
        """
        Forward pass for model.

        Parameters:
            context (torch.Tensor): The context tokens tensor of shape (batch_size, sequence_length).
            question (torch.Tensor): The question tokens tensor of shape (batch_size, sequence_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The start and end logits for answer predictions, each of shape (batch_size, sequence_length).
        """
        _,T = context.shape
        # Shape of context: (B,T)
        # Shape of question: (B,T)
        combined = torch.cat([context,question],dim=1)
        # Shape of combined: (B,2*T)
        embedded = self.embedding(combined)
        # Shape of embedded: (B,2*T,embedding_dim)
        lstm1_out, _ = self.lstm1(embedded)
        # Shape of lstm1_out: (B,T,2*hidden_dim)
        expert_output = self.expert(lstm1_out)
        # Shape of expert_output: (B,T,2*hidden_dim)
        lstm2_out, _ = self.lstm2(expert_output)
        # Shape of of lstm2_out: (B,T,2*hidden_dim)
        logits = self.output_linear(lstm2_out).squeeze(-1)
        # Shape of logits: (B,2*T)
        start_logits,end_logits = logits[:,:T],logits[:,T:]
        return start_logits,end_logits
        

class SquadLSTM_MoE(nn.Module):
    """
    An LSTM model for the SQuAD question answering task, enhanced with a Mixture of Experts (MoE) layer.

    Attributes:
        embedding (nn.Embedding): Embedding layer for token vector representations.
        lstm1 (nn.LSTM): The first LSTM layer for initial processing of combined context and question inputs.
        moe (SparseMoE): Sparse Mixture of Experts layer for selective expert processing.
        lstm2 (nn.LSTM): The second LSTM layer for further processing after the MoE layer.
        output_linear (nn.Linear): Linear layer for predicting start and end indices of answers in the context.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 intermediate_expert_dim: int,
                 model_state_dict: dict,
                 router: str,
                 num_experts: int,
                 top_k: int,
                 num_layers: int = 1,
                 ) -> None:
        """
        Initializes the SquadLSTM_MoE model with specified parameters and MoE configuration.

        Parameters:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of token embeddings.
            hidden_dim (int): Dimensionality of LSTM hidden states.
            intermediate_expert_dim (int): Dimensionality of the expert layer's hidden layer.
            model_state_dict (dict): State dictionary to initialize the embedding layer.
            router (str): Type of router used in the MoE layer.
            num_experts (int): Number of experts in the MoE layer.
            top_k (int): Number of top experts to consider in the MoE layer.
            num_layers (int): Number of layers in each LSTM. Defaults to 1.
        """
        super(SquadLSTM_MoE,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.embedding.load_state_dict(model_state_dict)
        for params in self.embedding.parameters():
            params.requires_grad = False
        self.lstm1 = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True,batch_first=True,num_layers=num_layers)
        
        self.moe = SparseMoE(router,2*hidden_dim,intermediate_expert_dim,num_experts,top_k=top_k)
        
        self.lstm2 = nn.LSTM(hidden_dim*2,hidden_dim,bidirectional=True,batch_first=True,num_layers=num_layers)
        
        self.output_linear = nn.Linear(hidden_dim*2,1)
        
    def forward(self,context,question):
        """
        Forward pass for the SQuAD question answering task with MoE enhancement.

        Processes combined context and question inputs with MoE routing to predict start and end indices for answers.

        Parameters:
            context (torch.Tensor): The context tokens tensor of shape (batch_size, sequence_length).
            question (torch.Tensor): The question tokens tensor of shape (batch_size, sequence_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The start and end logits for answer predictions, each of shape (batch_size, sequence_length).
        """
        _,T = context.shape
        # Shape of context: (B,T)
        # Shape of question: (B,T)
        combined = torch.cat([context,question],dim=1)
        # Shape of combined: (B,2*T)
        embedded = self.embedding(combined)
        # Shape of embedded: (B,2*T,embedding_dim)
        lstm1_out, _ = self.lstm1(embedded)
        # Shape of lstm1_out: (B,2*T,2*hidden_dim)
        moe_output = self.moe(lstm1_out)
        # Shape of moe_output: (B,2*T,2*hidden_dim)
        lstm2_out, _ = self.lstm2(moe_output)
        # Shape of of lstm2_out: (B,2*T,2*hidden_dim)
        logits = self.output_linear(lstm2_out).squeeze(-1)
        # Shape of logits: (B,2*T)
        start_logits,end_logits = logits[:,:T],logits[:,T:]
        return start_logits,end_logits