import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import clones
from model.TransformerModule import MultiHeadedAttention, LayerNorm, SublayerConnection, PositionwiseFeedForward


class RNNBaseEncoder(nn.Module):
    def __init__(self, cell_type,
                 input_size,
                 output_size,
                 num_layers,
                 bidirectional=False,
                 dropout=0.1):
        super(RNNBaseEncoder, self).__init__()
        assert cell_type in ['GRU', 'LSTM']

        if bidirectional:
            assert output_size % 2 == 0
            cell_size = output_size // 2
        else:
            cell_size = output_size

        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.rnn_cell = getattr(nn, cell_type)(input_size=input_size,
                                               hidden_size=cell_size,
                                               num_layers=num_layers,
                                               bidirectional=bidirectional,
                                               dropout=dropout,
                                               batch_first=True)

    def forward(self, x,  # [batch, seq, dim]
                length):  # [batch, ]
        x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)

        # output: [batch, seq,  directions * dim] 每个时间步的隐状态
        # final_state = [layers * directions, batch, dim] 每一层的最后一个状态，不管batch_first是true或false，batch都在中间
        output, final_state = self.rnn_cell(x)
        output = pad_packed_sequence(output, batch_first=True)[0]  # 返回output和length，不需要length了

        if self.bidirectional:
            if self.cell_type == 'GRU':
                final_state_forward = final_state[0::2, :, :]  # [layers, batch, dim] 偶数的地方是forward
                final_state_backward = final_state[1::2, :, :]  # [layers, batch, dim] 奇数的地方是backward
                final_state = torch.cat([final_state_forward, final_state_backward], 2)  # [layers, batch, 2 * dim]
            else:
                final_state_h, final_state_c = final_state
                final_state_h = torch.cat([final_state_h[0::2, :, :], final_state_h[1::2, :, :]], 2)  # [layers, batch, 2 * dim]
                final_state_c = torch.cat([final_state_c[0::2, :, :], final_state_c[1::2, :, :]], 2)  # [layers, batch, 2 * dim]
                final_state = (final_state_h, final_state_c)

        # output = [batch, seq, dim]
        # final_state = [layers, batch, directions * dim]
        return output, final_state


class TransformerEncoderLayer(nn.Module):
    """EncoderLayer is made up of self-attn and feed forward"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class TransformerEncoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(TransformerEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def make_transformer_encoder(d_model=512, N=6,
                             d_ff=2048, h=8, dropout=0.1):
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    encoder = TransformerEncoder(TransformerEncoderLayer(d_model, attn, ff, dropout), N)
    #  This was important from their code.
    for p in encoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return encoder
