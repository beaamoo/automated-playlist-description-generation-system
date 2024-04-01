import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from .ops import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, e_pos, max_length=100):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok_embedding = nn.Embedding(input_dim, hid_dim).to(self.device)
        self.pos_embedding = nn.Embedding(max_length, hid_dim).to(self.device)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout).to(self.device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)
        self.e_pos = e_pos

    def forward(self, src, src_mask):
        batch_size, src_len = src.shape[0], src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)
        
        src = self.tok_embedding(src) * self.scale.to(src.device)
        if self.e_pos:
            src += self.pos_embedding(pos)
        src = self.dropout(src)

        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src2 = self.self_attention(src, src, src, src_mask)[0]
        src = self.self_attn_layer_norm(src + self.dropout(src2))
        src2 = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(src2))
        return src

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, d_layers, n_heads, pf_dim, dropout, max_length=100):
        super().__init__()
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(d_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size, trg_len = trg.shape[0], trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(trg.device)
        
        trg = self.dropout((self.tok_embedding(trg) * self.scale.to(trg.device)) + self.pos_embedding(pos))
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.fc_out(trg)
        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention
class Transformer(LightningModule):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, True, max_length)
        self.decoder = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length)
        self.src_pad_idx = 0
        self.trg_pad_idx = 0
        self.save_hyperparameters()

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
    
    def training_step(self, batch, batch_idx):
        src, trg = batch
        output = self(src, trg[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(output, trg, ignore_index=self.trg_pad_idx)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)