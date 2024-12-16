import torch
from torch import nn
from torch.nn import functional as F
import math

class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model=128, max_seq_length=50):
        super().__init__()

        # Create a matrix of shape (max_seq_length, d_model)
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (won't be updated during backprop)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class LanguageTransformerModel(nn.Module):
    def __init__(self, num_languages=18, d_model=128, max_name_length=50):
        super().__init__()

        self.embeddings = nn.Embedding(65535, 128)
        self.positional_embeddings = PositionalEmbeddings(d_model, max_name_length)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model, 8)
        self.pooling_layers = nn.Sequential(
            nn.AvgPool1d(4),
            nn.AvgPool1d(4),
            nn.AvgPool1d(8)
        )
        self.projection = nn.Linear(20, 18)



    def forward(self, x):
        x = self.embeddings(x)
        x = self.positional_embeddings(x)
        x = self.transformer_encoder(x)
        x = self.pooling_layers(x)
        x = x.squeeze()
        x = F.softmax(self.projection(x), dim=-1)

        return x
    
def create_model_context():
    net = LanguageTransformerModel()
    crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    return net, crit, optimizer