import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# 

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda s: [itos[c] for c in s]

class BigramLanguageModel():
  def __init__(self):
    super().__init__()
    self.embed = nn.Embedding(vocab_size, vocab_size)

  def forward(self, x, y=None):
    logits = self.embed(x)
    if y==None:
      loss = None
    else:
      pass
    return logits, loss



