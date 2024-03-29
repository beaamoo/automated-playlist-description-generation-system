import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    """Implements a multi-head attention mechanism."""
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        # Ensure the hidden dimension is divisible by the number of heads
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads."
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads  # Dimension of each attention head
        
        # Linear layers for transforming the input queries, keys, and values
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        # Final linear layer to transform the concatenated output
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear transformation and splitting into n_heads
        Q = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.fc_k(key).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.fc_v(value).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(energy, dim=-1)
        
        # Apply attention to the values
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hid_dim)
        
        # Pass through the final linear layer
        x = self.fc_o(x)
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    """Implements a position-wise feedforward layer."""
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)  # First linear layer
        self.fc_2 = nn.Linear(pf_dim, hid_dim)  # Second linear layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Apply ReLU activation after the first linear layer and dropout
        x = self.dropout(torch.relu(self.fc_1(x)))
        # Return the output of the second linear layer
        x = self.fc_2(x)
        return x
