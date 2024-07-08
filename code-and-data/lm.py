from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def batch_to_labeled_samples(batch: torch.IntTensor) -> [torch.IntTensor, torch.IntTensor]:
    # The batches that we get from the reader have corpus-sequences of length max-context + 1.
    # We need to translate them to input/output examples, each of which is shorter by one.
    # That is, if our input is of dimension (b x n) our output is two tensors, each of dimension (b x n-1)
    inputs = batch[:,:-1]
    labels = batch[:,1:]
    return (inputs, labels)

def compute_loss(logits, gold_labels, ignore_index=0):
    # logits size is (batch, seq_len, vocab_size)
    # gold_labels size is (batch, seq_len)
    # NOTE remember to handle padding (ignore them in loss calculation!)
    # NOTE cross-entropy expects other dimensions for logits
    # NOTE you can either use cross_entropy from PyTorch, or implement the loss on your own.

    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)     # (batch * seq_len, vocab_size)
    gold_labels = gold_labels.reshape(-1)    # (batch * seq_len)
    
    # Compute the loss using cross_entropy, ignoring the padding token (if applicable)
    loss = F.cross_entropy(logits, gold_labels, ignore_index=ignore_index)
    
    return loss
