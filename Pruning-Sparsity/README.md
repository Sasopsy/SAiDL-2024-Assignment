# Pruning and Sparsity
<figure style="text-align: center;">
  <img src="report/report_images/fine_pruning.png" style="margin: 0 auto; display: block;"/>
  <figcaption>Finegrained Pruning</figcaption>
</figure>

## Report
The **final report** of this assignment can be found in **[here](report/report.pdf)**.

## Notebooks
- [`notebook.ipynb`](notebook.ipynb) entails the initial training, pruning, sensitivity scanning and finetuning of our models.
- [`bonus_task.ipynb`](bonus_task.ipynb) is the bonus task of this assignment where I plot the weight distributions of our models and their activations. 

## Models
Code to the model used can be found in [`model.py`](model.py). The weights of the models can be found [here](https://drive.google.com/drive/folders/1jaYqiK3PmFSqR_bk5WMhudv1FeUAJK0M?usp=sharing). 

To load the weights of the unpruned model, run the following script:
```python 3
import torch
from model import VGG16

model = VGG16(10,(32,32))
weights = torch.load("/path/to/model")
model.load_state_dict(weights)
```
To load the weights of the pruned models, run the following:
```python 3
import torch
from model import VGG16

model = VGG16(10,(32,32))
weights = torch.load("/path/to/model")['model_state_dict']
model.load_state_dict(weights)
```
To load the pruning masks for any sort of further fine-tuning, run the following snippet:
```python 3
import torch

mask = torch.load("/path/to/model")['prune_masks']
```
## Trainer 
The code to the custom model trainer and fine-tuner can be found in [`trainer.py`](trainer.py).

## Pruners
[`pruner.py`](pruner.py) contains code for fine-grained, kernel and filter pruning strategies. They prune a given model and even generates a mask for them.

## Sensitivity Scan
[`scanner.py`](scanner.py) contains the code for performing systematic sensitivity scans for a given pruning strategy and model.

## Utils
[`utils.py`](utils.py) contains code for some useful utility functions (also contains the starter code provided by SAiDL).