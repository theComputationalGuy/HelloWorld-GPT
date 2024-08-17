import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

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
    if y==None:
      loss = None
    else:
      logits, loss = 



