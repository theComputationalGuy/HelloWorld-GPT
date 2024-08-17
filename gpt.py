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

class Head(nn.Module):
  def __init__(self, head_size) -> None:
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    wt = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
    wt = wt.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wt = F.softmax(wt)
    wt = self.dropout(wt)
    out = wt @ v
    
    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size) -> None:
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self):
    out = torch.cat([h(x) for h in self.heads])
    out = self.proj(out)
    return out


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



