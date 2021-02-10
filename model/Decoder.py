import copy
import torch.nn as nn
import torch.nn.functional as F
from utils import clones
from model.TransformerModule import MultiHeadedAttention, LayerNorm, SublayerConnection, PositionwiseFeedForward


class RNNBaseDecoder(nn.Module):
    def __init__(self, cell_type,
                 input_size,
                 output_size,
                 num_layers,
                 dropout=0.1
                 ):
        super(RNNBaseDecoder, self).__init__()
        assert cell_type in ['GRU', 'LSTM']

        self.cell_type = cell_type
        self.rnn_cell = getattr(nn, cell_type)(input_size=input_size,
                                               hidden_size=output_size,
                                               num_layers=num_layers,
                                               dropout=dropout,
                                               batch_first=True)

    def forward(self, x,  # x = [batch, seq, dim] 或单步输入 [batch, 1, dim]
                state):  # state = [layers * directions, batch, dim]
        output, final_state = self.rnn_cell(x, state)
        return output, final_state


class TransformerDecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class TransformerDecoder(nn.Module):
    """Generic N layer decoder with masking."""
    def __init__(self, layer, N):
        super(TransformerDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


def make_transformer_decoder(d_model=512, N=6,
                             d_ff=2048, h=8, dropout=0.1):
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    decoder = TransformerDecoder(TransformerDecoderLayer(d_model, attn, copy.deepcopy(attn), ff, dropout), N)
    #  This was important from their code.
    for p in decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return decoder


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
