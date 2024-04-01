import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer
import pandas as pd
import re

class ConllDataset(Dataset):
    """
    A PyTorch Dataset class for handling CoNLL formatted data for Named Entity Recognition (NER) tasks.
    
    This dataset class handles tokenization of text sequences and prepares them for training or evaluating NER models,
    including padding of sequences and NER tags to uniform lengths.
    
    Attributes:
        data (list of dicts): The dataset, where each item is a dictionary with keys like 'tokens' and 'ner_tags'.
        tokenizer (TokenizerConll): An instance of TokenizerConll used for tokenizing text sequences.
        label_padding_value (int): The value used for padding the NER tags. Defaults to -100.
    """
    def __init__(self, data,
                 tokenizer: Tokenizer,
                 label_padding_value=-100):
        """
        Initializes the ConllDataset with the dataset, tokenizer, and label padding value.
        
        Parameters:
            data (list of dicts): The dataset containing tokens and corresponding NER tags.
            tokenizer (TokenizerConll): The tokenizer for encoding the text sequences.
            label_padding_value (int): The padding value for NER tags. Defaults to -100.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label_padding_value = label_padding_value
    
    def __len__(self):
        """
        Returns the number of items in the dataset.
        
        Returns:
            int: The dataset size.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index in the dataset.
        
        Parameters:
            idx (int): The index of the item.
            
        Returns:
            tuple: A tuple containing the encoded tokens tensor and the padded NER tags tensor.
        """
        item = self.data[idx]
        tokens = item['tokens']
        ner_tags = item['ner_tags']
        
        # Encode tokens using the provided tokenizer
        encoded_tokens = self.tokenizer.encode(tokens)
        
        # Pad ner_tags to the same length as encoded_tokens
        padded_ner_tags = self._pad_labels(ner_tags)
        
        return encoded_tokens, padded_ner_tags
    
    def _pad_labels(self,labels):
        """
        Pads the labels to the maximum length specified in the tokenizer.
        
        Parameters:
            labels (list of int): The NER tags for a single sequence.
        
        Returns:
            torch.Tensor: A tensor of padded NER tags.
        """
        padded_labels = labels + [self.label_padding_value] * (self.tokenizer.maxlen - len(labels))
        return torch.tensor(padded_labels, dtype=torch.long)


class SquadDataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing the SQuAD dataset for question answering tasks. 
    
    This dataset class handles tokenization of text sequences and prepares them for training or evaluating  Qand A models,
    including padding of to uniform lengths and calculating end of answer end.

    Attributes:
    - dataframe (pd.DataFrame): A DataFrame containing the SQuAD dataset with columns for context, question, and answer, answer_start and answet_start_token.
    - tokenizer (Tokenizer): An instance of a tokenizer that is compatible with the dataset's text.
    - max_len (int): The maximum length of the tokenized input.
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 tokenizer: Tokenizer,
                 max_len=256):
        """
        Initializes the SquadDataset with a dataframe, a tokenizer, and an optional maximum sequence length.

        Args:
        - dataframe (pd.DataFrame): The dataframe containing the SQuAD dataset.
        - tokenizer (Tokenizer): The tokenizer to use for encoding the text.
        - max_len (int, optional): The maximum length of a sequence after tokenization. Defaults to 256.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataframe = dataframe  
        self.regex = r'\w+|[^\w\s]'
    
    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
        - int: The total number of rows in the dataframe.
        """
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Retrieves an item by index from the dataset and returns tokenized context, question, and the answer span.

        Args:
        - idx (int): The index of the item.

        Returns:
        - tuple[torch.Tensor]: A tuple containing tokenized context, question, and the start and end token indices of the answer.
        """
        row = self.dataframe.iloc[idx]
        context = str(row['context']).lower()
        question = str(row['question']).lower()
        answer_text = str(row['answer_text']).lower()
        answer_start = row['answer_start_token_index']
        answer_text_tokens = re.findall(self.regex,answer_text)
        answer_end = answer_start + len(answer_text_tokens) - 1
        
        # Tokenize context and question
        context_tokens = re.findall(self.regex, context)
        question_tokens = re.findall(self.regex, question)
        
        # Encode the context and the question using the tokenizer
        encoded_context = self.tokenizer.encode(context_tokens)
        encoded_question = self.tokenizer.encode(question_tokens)
        
        return encoded_context,encoded_question,answer_start,answer_end
        