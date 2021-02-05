import torch.nn as nn
import copy
from model.TransformerModule import PositionalEncoding
from model.Encoder import make_transformer_encoder
from model.Decoder import make_transformer_decoder, Generator
from model.EncoderDecoder import Transformer
from model.Embedding import Embedding


def make_model(src_vocab, tgt_vocab, N=6,
        d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    position = PositionalEncoding(d_model, dropout)
    encoder = make_transformer_encoder(d_model, N, d_ff, h, dropout)
    decoder = make_transformer_decoder(d_model, N, d_ff, h, dropout)
    model = Transformer(
            encoder=encoder,
            decoder=decoder,
            src_embed=nn.Sequential(Embedding(num_vocab=src_vocab, embedding_size=d_model), c(position)),
            tgt_embed=nn.Sequential(Embedding(num_vocab=tgt_vocab, embedding_size=d_model), c(position)),
            generator=Generator(d_model=d_model, vocab=tgt_vocab)
            )
    #  This was important from their code.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

    
#  if __name__ == '__main__':
#      tmp_model = make_model(10, 10, 2)
#      print(tmp_model)
    







