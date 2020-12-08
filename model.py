import torch
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp_models.generation import SimpleSeq2Seq
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper


def create_seq2seqmodel(vocab, embedding_dim=100, hidden_dim=100, 
                        encoder=None, max_decoding_steps=20, beam_size=1, use_bleu=True):
    embedding_dim = embedding_dim
    hidden_dim = hidden_dim
    embedding = Embedding(embedding_dim=embedding_dim,
                          num_embeddings=vocab.get_vocab_size("tokens"))
    source_embedder = BasicTextFieldEmbedder({'tokens': embedding})
    encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True))
    max_decoding_steps = max_decoding_steps
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=embedding_dim, beam_size=beam_size,
                          use_bleu=use_bleu)

    return model