import torch

class Tokenizer:
    """
    A tokenizer class for tokenizing and encoding sequences with padding.
    
    Attributes:
        idx2token (dict): A dictionary mapping indices to tokens.
        token2idx (dict): A dictionary mapping tokens to indices.
        maxlen (int): The maximum length to which the sequences will be padded or truncated.
        padding_mode (str): The mode of padding, either 'pre' for pre-padding or 'post' for post-padding.
        padding_value (int): The value used for padding.
    """  
    def __init__(self,
                 idx2token: dict,
                 token2idx: dict,
                 maxlen: int = 128,
                 padding_mode: str = 'post',
                 padding_value = 0) -> None:
        """
        Initializes the Tokenizer class.
        
        Parameters:
            idx2token (dict): Mapping from indices to tokens.
            token2idx (dict): Mapping from tokens to indices.
            maxlen (int): Maximum length of sequences after padding.
            padding_mode (str): Padding strategy ('pre' or 'post').
            padding_value (int): Value to use for padding.
        """
        assert padding_mode in ['post', 'pre'], f"No padding option as {padding_mode}, only post and pre are available."
        self.idx2token = idx2token
        self.idx2token = idx2token
        self.token2idx = token2idx
        self.maxlen = maxlen
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        
    def pad_sequences(self,
                      input_seq: list[list[int]] | list[int]) -> torch.Tensor:
        """
        Pads sequences to a specified maximum length.
        
        Parameters:
            input_seq (list[list[int]] | list[int]): A list of sequences or a single sequence to be padded.
            
        Returns:
            torch.Tensor: A tensor of padded sequences - shape (B,C) for list of sequences and shape (C) for a single sequence.
        """
        # Determine if input_seq is a single sequence or a list of sequences
        if all(isinstance(item, list) for item in input_seq):
            # Nested list (list of sequences)
            sequences = input_seq
        else:
            # Single sequence (flat list)
            sequences = [input_seq]
        
        # Create an empty tensor to hold the padded sequences
        padded_sequences = torch.full((len(sequences), self.maxlen), self.padding_value, dtype=torch.long)
        
        # Pad each sequence according to the specified padding style
        for i, seq in enumerate(sequences):
            length = min(len(seq), self.maxlen)
            if self.padding_mode == 'pre':
                padded_sequences[i, -length:] = torch.tensor(seq[:length], dtype=torch.long)
            else:  # Default to post-padding
                padded_sequences[i, :length] = torch.tensor(seq[:length], dtype=torch.long)
        
        padded_sequences = padded_sequences.squeeze()
        
        return padded_sequences
    
    def encode(self, 
               input_batch: list[str]) -> torch.Tensor:
        """
        Encodes a batch of text sequences into a tensor of token indices, applying padding as necessary.
        
        Parameters:
            input_batch (list[str]): A batch of text sequences.
            
        Returns:
            torch.Tensor: A tensor representing the encoded and padded sequences.
        """
        # Lower the letters
        input_batch = list(map(lambda x: x.lower(),input_batch))
        
        indexed_batch = []
        indexed_seq = [self.token2idx.get(token, self.token2idx['<UNK>']) for token in input_batch]  # Use '<UNK>' for unknown tokens
        indexed_batch.append(indexed_seq)
        
        return self.pad_sequences(indexed_batch)