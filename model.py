import torch
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp_models.generation import SimpleSeq2Seq, Bart
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, LstmSeq2SeqEncoder


def create_seq2seqmodel(vocab, embedding_dim=100, hidden_dim=100, 
                        encoder=None, max_decoding_steps=20, beam_size=1, use_bleu=True):
    device = torch.device('cuda:0')
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
    # model = Bart(model_name='facebook/bart-base', vocab=vocab, max_decoding_steps=20, beam_size=1)
    model.to(device)
    return model
