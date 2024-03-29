import torch
import torch.nn as nn
from .ops import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer

class Encoder(nn.Module):
    """Defines an Encoder block for a Transformer model."""
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, e_pos, device, max_length=64):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.e_pos = e_pos

    def forward(self, src, src_mask):
        """Forward pass for the Encoder block."""
        batch_size, src_len = src.shape[0], src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        src = self.tok_embedding(src) * self.scale
        if self.e_pos:
            src += self.pos_embedding(pos)
        src = self.dropout(src)

        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class EncoderLayer(nn.Module):
    """Defines an Encoder layer within the Encoder block."""
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """Forward pass for the Encoder layer."""
        src2 = self.self_attention(src, src, src, src_mask)[0]
        src = self.self_attn_layer_norm(src + self.dropout(src2))
        src2 = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(src2))
        return src

class Decoder(nn.Module):
    """Defines a Decoder block for the Transformer model."""
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=64):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """Forward pass for the Decoder block."""
        batch_size, trg_len = trg.shape[0], trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)
        return output, attention

class DecoderLayer(nn.Module):
    """Defines a Decoder layer within the Decoder block."""
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """Forward pass for the Decoder layer, including self-attention and encoder-decoder attention."""
        # Self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # Encoder-decoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # Positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention

class Transformer(nn.Module):
    """Defines the Transformer model, containing both the Encoder and Decoder blocks."""
    def __init__(self, input_size, output_size, hidden_size, e_layers, d_layers, heads, pf_dim, dropout, e_pos, device):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, e_layers, heads, pf_dim, dropout, e_pos, device)
        self.decoder = Decoder(output_size, hidden_size, d_layers, heads, pf_dim, dropout, device)
        self.device = device
        # Assume src_pad_idx and trg_pad_idx are known or provided
        self.src_pad_idx = 0
        self.trg_pad_idx = 0

    def make_src_mask(self, src):
        """Creates a mask for the source input."""
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        """Creates a mask for the target input."""
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        return trg_pad_mask & trg_sub_mask

    def forward(self, src, trg):
        """Forward pass for the Transformer model."""
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention
