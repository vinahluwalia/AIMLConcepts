


# Encoder Block
# Decoder Block

# Positional Encoding
# Component of the trasnformer responsible for capturing information about the position of each token in the sequence


# Attention Mechanisms
# Self Attention

# Multi-Head Attention
# - Splits inout into multiple heads
# - Each head has different weights

# Position-wise feed-forward networks

import torch.nn as nn

model= nn.Transformer(
    d_model = 512, # Dimensionality of model inputs
    nhead = 8, # Number of attention heads
    num_encoder_layers = 6, # Number of encoder layers
    num_decoder_layers = 6, # Number of decoder layers
)

print(model)

# Transformer model with 8 attention heads, 6 encoder and decoder layers, and for input sequence embeddings of length 1536

# Transformer(
#   (encoder): TransformerEncoder(
#     (layers): ModuleList(
#       (0-5): 6 x TransformerEncoderLayer(
#         (self_attn): MultiheadAttention(
#           (out_proj): NonDynamicallyQuantizableLinear(in_features=1536, out_features=1536, bias=True)
#         )
#         (linear1): Linear(in_features=1536, out_features=2048, bias=True)
#         (dropout): Dropout(p=0.1, inplace=False)
#         (linear2): Linear(in_features=2048, out_features=1536, bias=True)
#         (norm1): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
#         (norm2): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
#         (dropout1): Dropout(p=0.1, inplace=False)
#         (dropout2): Dropout(p=0.1, inplace=False)
#       )
#     )
#     (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
#   )
#   (decoder): TransformerDecoder(
#     (layers): ModuleList(
#       (0-5): 6 x TransformerDecoderLayer(
#         (self_attn): MultiheadAttention(
#           (out_proj): NonDynamicallyQuantizableLinear(in_features=1536, out_features=1536, bias=True)
#         )
#         (multihead_attn): MultiheadAttention(
#           (out_proj): NonDynamicallyQuantizableLinear(in_features=1536, out_features=1536, bias=True)
#         )
#         (linear1): Linear(in_features=1536, out_features=2048, bias=True)
#         (dropout): Dropout(p=0.1, inplace=False)
#         (linear2): Linear(in_features=2048, out_features=1536, bias=True)
#         (norm1): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
#         (norm2): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
#         (norm3): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
#         (dropout1): Dropout(p=0.1, inplace=False)
#         (dropout2): Dropout(p=0.1, inplace=False)
#         (dropout3): Dropout(p=0.1, inplace=False)
#       )
#     )
#     (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
#   )
# )

# Embedding and positional encoding
# Tokens >> Embedding Vectors

import torch
import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int): 
        # Vocabulary size pertains to the tokenizer's vocabulary size
        # d_model: Dimensionality of the model or the dimnesions of the input embeddings
        super().__init__()
        self.d_model = d_model # Dimensionallity of the input embeddings
        self.vocab_size = vocab_size # Model vocabulary size
        self.embedding = nn.Embedding(vocab_size, d_model) # Embedding layer to map each token in the vocabulary to a d_model dimensional vector
    
    def forward(self, x): # Calculates and returns the embeddings
        return self.embedding(x) * math.sqrt(self.d_model) # Scaling the embeddings by square root of d_model ensures that token embeddings
        # don't overwhelm or be overwhelmbed by positional embeddings
    
# Create embeddings
# Embedding Layer of dimnensionality 512 and vocab size of 10000
embedding_layer = InputEmbeddings(vocab_size=10000, d_model=512)
# Pass an example batch of two sequences
# Each sequence with four token IDs
batch = torch.tensor([[1,2,3,4], [5,6,7,8]])
embedding_output = embedding_layer(batch)
embedding_output.shape
# torch.Size([2, 4, 512]) # 2 Sequences, 4 Tokens, 512 Dimensions

# Positional Encoding
# Positional encoding encodes each tokens position in the sequence into a positional embedding
# Adds them to the token embeddings to capture the positional information
# Token and Positional embeddings have the same dimensionality for easier addition
# Positional Encoding is calculated using sine and cosine functions of different frequencies
# Sine is used for even dimensions and cosine for odd dimensions
# Positional Encoding is added to the token embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length):

        super().__init__()
        pe = torch.zeros(max_seq_length, d_model) # Initialize a positional embedding -> Total number of sequences and dimensions of the model
        # Tensor of positions for each token in the sequence
        position = torch.arange(0, max_seq_length, dtype = torch.float).unsqueeze(1) # Unsqueeze dim = 1, gives columnar shape
        # Sine and Cosine calculations
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype = torch.float) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # Sine for even dimensions
        pe[:, 1::2] = torch.cos(position * div_term) # Cosine for odd dimensions

        # Ensure pe is not a learnable parameter
        self.pe = self.register_buffer('pe', pe.unsqueeze(0)) # Register buffer so it's not considered a model parameter



        




