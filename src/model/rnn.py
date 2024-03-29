import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class Encoder(nn.Module):
    """Defines an encoder with GRU layer for the sequence to sequence model."""
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)
        outputs = (outputs[:, :, :hidden.size(2)] + outputs[:, :, hidden.size(2):])  # Sum bidirectional outputs
        return outputs, hidden

class Attention(nn.Module):
    """Implements the attention mechanism."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # Prepare for batch processing
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)

class Decoder(nn.Module):
    """Defines a decoder with GRU layer and attention for the sequence to sequence model."""
    def __init__(self, embed_size, hidden_size, output_size, n_layers=1, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        embedded = self.embed(input).unsqueeze(0)
        embedded = self.dropout(embedded)
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], dim=1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

class RNN_Attn(nn.Module):
    """Integrates Encoder, Decoder into a sequence to sequence model with attention."""
    def __init__(self, input_size, output_size, embed_size, hidden_size, e_layers, d_layers, dropout, teacher_forcing_ratio):
        super().__init__()
        self.encoder = Encoder(input_size, embed_size, hidden_size, e_layers, dropout)
        self.decoder = Decoder(embed_size, hidden_size, output_size, d_layers, dropout)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, trg):
        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = trg.data[0, :]  # sos token
        outputs = Variable(torch.zeros(trg.size(0), trg.size(1), self.decoder.output_size))
        for t in range(1, trg.size(0)):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < self.teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = trg.data[t] if is_teacher else top1
        return outputs
