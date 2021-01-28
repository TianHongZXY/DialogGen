import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import clones


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    For code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))  # 这么写应该是原论文的意思


def attention(query, key, value, mask=None, dropout=None):
    """计算 Scaled Dot Product Attention"""
    #  query shape = [nbatches, h, T_q, d_k]
    #  key shape = value shape = [nbatches, h, T_k, d_k(d_v)] 默认假设了d_v == d_k
    d_k = query.size(-1)
    # scores shape = p_attn shape = [nbatches, h, T_q, T_k]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # src / tgt mask shape = [nbatches, 1, 1, T_k] / [nbatches, 1, T_q, T_q]
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # 必须d_k == d_q, 同时默认假设d_v == d_k(也可以不相等)，因此d_k == d_v == d_q
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        assert query.size(-1) == key.size(-1) == self.d_model
        # src / tgt mask shape = [nbatches, 1, T_k] / [nbatches, T_q, T_q]
        if mask is not None:
            # src / tgt mask shape = [nbatches, 1, 1, T_k] / [nbatches, 1, T_q, T_q]
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        #  1) Do all the linear projections in batch from d_model => h * d_k
        #  q k v shape = [nbatches, h, T_q(T_k), d_k(d_v)], h * d_k == d_model
        query, key, value = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # 先把最后一维d_model拆成h * d_k，再转置
                for l, x in zip(self.linears, (query, key, value))]

        #  2) Apply attention on all the projected vectors in batch.
        #  x shape = [nbatches, h, T_q, d_k], attn shape = [nbatches, h, T_q, T_k]
        x, self.attn = attention(query, key, value, mask, self.dropout)
        #  3) Concat using a view and apply a final linear
        x = x.transpose(1, 2).contiguous()\
                .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                            -(math.log(1e4) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #  pe shape = [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #  x shape = [nbatches, T, d_model]
        #  pe[:, :x.size(1)] shape = [1, T, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
