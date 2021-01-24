import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import numpy as np
import math


def clones(module, N):
    "生成N个同样的layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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


def subsequent_mask(size):
    "将后续部分mask掉"
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "计算 Scaled Dot Product Attention"
    #  query shape = [nbatches, h, T_q, d_k]
    #  key shape = value shape = [nbatches, h, T_k, d_k(d_v)] 默认假设了d_v == d_k
    d_k = query.size(-1)
    # scores shape = p_attn shape = [nbatches, h, T_q, T_k]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # src / tgt mask shape = [nbatches, 1, 1, T_k] / [nbatches, 1, T_q, T_q]
        scores.masked_fill(mask == 0, -1e9)
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


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * 
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
        return x + self.dropout(sublayer(self.norm(x)))
        #  return self.norm(x + self.dropout(sublayer(x)))  # 这么写应该是原论文的意思


class EncoderLayer(nn.Module):
    "EncoderLayer is made up of self-attn and feed forward"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size =size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6,
        d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
            encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            src_embed=nn.Sequential(Embedding(d_model, src_vocab), c(position)),
            tgt_embed=nn.Sequential(Embedding(d_model, tgt_vocab), c(position)),
            generator=Generator(d_model, tgt_vocab)
            )
    #  This was important from their code.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

    
#  if __name__ == '__main__':
#      tmp_model = make_model(10, 10, 2)
#      print(tmp_model)
    







