from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTERPRET_MODE = False
stop = 0
attention_weights_l = []

def plot_attention_weights(output_file="attention_plots.pdf"):
    with PdfPages(output_file) as pdf:
        for attention_weights in attention_weights_l:
            for attention_weight in attention_weights:
                plt.figure(figsize=(5, 5))
                sns.heatmap(attention_weight.detach().cpu().numpy(), cmap='viridis')
                plt.title(f'Attention Heatmap')
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                pdf.savefig()  # Save the current figure into the PDF
                plt.close()  # Close the figure to avoid display

def create_kqv_matrix(input_vector_dim, n_heads = 1):
    new_dim = int(input_vector_dim / n_heads)
    return nn.Linear(input_vector_dim, 3 * new_dim) # TODO fill in the correct dimensions


def kqv(x, linear):
    B, N, D = x.size()
    kqv_x = linear(x)
    k, q, v = kqv_x.chunk(3, dim=2)
    return k, q, v


def kqv_test():
    # Example input tensor
    batch_size = 2
    sequence_length = 10
    input_vector_dim = 512
    x = torch.randn(batch_size, sequence_length, input_vector_dim).to(device)
    kqv_matrix = create_kqv_matrix(input_vector_dim, 2).to(device)
    k, q, v = kqv(x, kqv_matrix)
    print(f'K shape: {k.shape}, Q shape: {q.shape}, V shape: {v.shape}')


def attention_scores(a, b):

    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    # Compute the dot product between a and b.T (transpose last two dimensions of b)
    A = torch.bmm(a, b.transpose(1, 2))

    # Scale the dot products by the square root of the dimensionality
    A = A / torch.sqrt(torch.tensor(D1, dtype=torch.float32, device=device))

    return A


def create_causal_mask(embed_dim, n_heads, max_context_len):
    # Return a causal mask (a tensor) with zeroes in dimensions we want to zero out.
    # This function receives more arguments than it actually needs. This is just because
    # it is part of an assignment, and I want you to figure out on your own which arguments
    # are relevant.

    # Create an n x n lower triangular matrix
    n = max_context_len
    mask = torch.tril(torch.ones((1, n, n), dtype=torch.float32, device=device))
    return mask


def self_attention(v, A, mask = None):
    global INTERPRET_MODE

    B, N, D = A.size()

    # Slice mask to create a 1 x N x N tensor
    mask = mask[:, :N, :N]
    M = mask.repeat(B, 1, 1)
    A = A.masked_fill(M == 0, float("-inf"))

    # compute sa (corresponding to y in the assignemnt text).
    # As usual, the dimensions of v and of sa are (b x n x d).
    #norm_A = F.softmax(A, dim=1)
    norm_A = F.softmax(A, dim=-1)

    if INTERPRET_MODE:
        global stop
        if stop < 36:
            attention_weights_l.append(norm_A)
            stop += 1
        else:
            plot_attention_weights()
            INTERPRET_MODE = False

    # Compute the dot product between norm_A and v
    sa = torch.bmm(norm_A, v)
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(q, k)
    sa = self_attention(v, att, attention_mask)
    return sa


def multi_head_attention_layer(x, kqv_matrices, mask):
    B, N, D = x.size()
    sa = self_attention_layer(x, kqv_matrices[0], mask)
    if len(kqv_matrices) > 1:
        for kqv_matrix in kqv_matrices[1:]:
            new_sa = self_attention_layer(x, kqv_matrix, mask)
            sa = torch.cat((sa, new_sa), dim=2)


    # There is also a tricker (but more efficient) version of multi-head attention, where we do all the computation
    # using a single multiplication with a single kqv_matrix (or a single kqv_tensor) and re-arranging the results afterwards.
    # If you want a challenge, you can try and implement this. You may need to change additional places in the code accordingly.
    assert sa.size() == x.size()
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads).to(device) for i in range(n_heads)])
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len).to(device)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim).to(device)

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        sa = self.proj(sa)
        return sa

# TEST
kqv_test()