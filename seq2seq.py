import torch.nn as nn
import torch
import random
import torch.nn.functional as F
from itertools import chain
from torchsnooper import snoop as torchsnoop


class Attention(nn.Module):
    """加法attention"""
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim]
        # mask = [batch size, src len], <pad> token position is 0 otherwise 1
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        # hidden = [batch size, src len, dec hid dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # encoder_outputs = [batch size, src len, enc hid dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # energy = [batch size, src len, dec hid dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # attention= [batch size, src len]
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, 1e-10)

        return F.softmax(attention, dim=1)


class Classifier2layer(nn.Module):
    def __init__(self, input_dim, num_class):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = num_class
        self.classifer = nn.Linear(in_features=input_dim, out_features=num_class)

    def forward(self, x):
        outputs = self.classifer(x)
        return outputs
        # return F.softmax(outputs, dim=1)


class Conv1dEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 kernel_size,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd!"

        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [0, 1, 2, 3, ..., src len - 1]

        # pos = [batch size, src len]

        # embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)

        # tok_embedded = pos_embedded = [batch size, src len, emb dim]

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)

        # embedded = [batch size, src len, emb dim]

        # pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)

        # conv_input = [batch size, src len, hid dim]

        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)

        # conv_input = [batch size, hid dim, src len]

        # begin convolutional blocks...

        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            # conved = [batch size, 2 * hid dim, src len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, src len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim, src len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        # ...end convolutional blocks

        # permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))

        # conved = [batch size, src len, emb dim]

        # elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale

        # combined = [batch size, src len, emb dim]

        return conved, combined


class ConvEncoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, output_dim, kernel_size, stride, dropout=0, sent_len3=5):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hid_dim, kernel_size=(kernel_size, self.emb_dim), stride=stride)
        nn.init.uniform_(self.conv1.weight, -0.001, 0.001)
        self.conv2 = nn.Conv2d(in_channels=hid_dim, out_channels=2 * hid_dim, kernel_size=(kernel_size, 1), stride=stride)
        self.conv3 = nn.Conv2d(in_channels=2 * hid_dim, out_channels=output_dim, kernel_size=(sent_len3, 1), stride=stride)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=2 * hid_dim, out_features=output_dim)

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs):
        """
        Inputs must be embeddings: batch_size x seq_len x emb_dim
        """
        inputs = self.dropout(inputs.unsqueeze(1))
        # inputs = [batch_size, 1, seq_len, emb_dim]
        x1 = F.relu(self.conv1(inputs))
        x1 = self.dropout(x1)
        # x1 = [batch_size, output_dim, new_seq_len, 1]
        x2 = F.relu(self.conv2(x1))
        x2 = self.dropout(x2)
        # x2 = [batch_size, output_dim, new_seq_len', 1]
        # x3 = F.tanh(self.conv3(x2).squeeze())
        x3 = self.conv3(x2).squeeze()
        # outputs = F.sigmoid(self.fc(x3))
        # outputs = [batch_size, output_dim, 1, 1].squeeze()
        # x1 = F.max_pool1d(x1, x1.size(2)).squeeze()
        # [batch_size, hid_dim]
        # x2 = F.max_pool1d(x2, x2.size(2)).squeeze()
        # x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        # batch_size x (3 * hid_dim)
        # outputs = F.sigmoid(self.fc(torch.cat([x1, x2, x3], dim=1)))
        # outputs = self.fc(torch.cat([x1, x2, x3], dim=1))

        return x3


class Discriminator(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx, feature_dim, n_filters, kernel_size, stride, sent_len3, dropout=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=padding_idx)
        self.topic_encoder = ConvEncoder(emb_dim=emb_dim, hid_dim=n_filters, output_dim=feature_dim,
                                         kernel_size=kernel_size, stride=stride, dropout=dropout, sent_len3=sent_len3)
        self.persona_encoder = ConvEncoder(emb_dim=emb_dim, hid_dim=n_filters, output_dim=feature_dim,
                                           kernel_size=kernel_size, stride=stride, dropout=dropout, sent_len3=sent_len3)

    def forward(self, src, trg):
        src = self.embedding(src)
        trg = self.embedding(trg)
        trg_t_feature = self.topic_encoder(trg)
        trg_p_feature = self.persona_encoder(trg)
        src_t_feature = self.topic_encoder(src)
        src_p_feature = self.persona_encoder(src)

        return {'src_p_feature': src_p_feature, 'src_t_feature': src_t_feature,
                'trg_p_feature': trg_p_feature, 'trg_t_feature': trg_t_feature}

# class FeatureExtractor(nn.Module):
#     def __init__(self, encoder):
#         super().__init__()
#         self.encoder = encoder
#
    # def forward(self, s, t):
    #     # s = t = [batch_size, seq_len, emb_dim]
    #     s_feature = self.encoder(s)
    #     # [batch_size, 3 * hid_dim]
    #     t_feature = self.encoder(t)
    #     # [batch_size, 1, 1]
    #     M = F.sigmoid(torch.bmm(s_feature.unsqueeze(1), t_feature.unsqueeze(1).T))
    #     return M.squeeze(), s_feature, t_feature
    #
    # def get_output_dim(self):
    #     return self.encoder.output_dim


class Generator(nn.Module):
    def __init__(self, vocab_size, emb_dim, context_encoder, discriminator,
                 agg_output_dim, lstm_input_dim, lstm_hid_dim, n_layers=1, dropout=0,
                 padding_idx=None, K=1, device='cpu'
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.discriminator = discriminator
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        # self.topic_encoder = discriminator.topic_encoder
        # self.persona_encoder = discriminator.persona_encoder
        self.context_encoder = context_encoder
        self.feature_dim = self.discriminator.topic_encoder.get_output_dim()
        self.agg_output_dim = agg_output_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hid_dim, num_layers=n_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.AGG = nn.Linear(in_features=K * self.context_encoder.get_output_dim(),
                             out_features=self.agg_output_dim)
        self.fc_H0 = nn.Linear(in_features=self.agg_output_dim + 2 * self.feature_dim,
                               out_features=lstm_hid_dim)
        self.fc_pred = nn.Linear(in_features=lstm_hid_dim, out_features=vocab_size, bias=False)

        self.encoder_decoder = nn.ModuleList([self.embedding, self.context_encoder,
                                              self.AGG, self.fc_H0, self.rnn, self.fc_pred]
                                             )

    # @torchsnoop()
    def forward_encoder(self, src, trg):
        src_embed = self.embedding(src.T)
        context = self.context_encoder(src_embed)
        # [batch_size, agg_output_dim]
        C = self.AGG(context)
        # [batch_size, feature_dim]
        # trg_t_feature = self.topic_encoder(trg)
        # trg_p_feature = self.persona_encoder(trg)
        feature_dict = self.discriminator(src.T, trg.T)
        trg_t_feature = feature_dict['trg_t_feature']
        trg_p_feature = feature_dict['trg_p_feature']
        H0 = self.fc_H0(torch.cat([C, trg_t_feature, trg_p_feature], dim=1))
        # H0 = [1, batch_size, lstm_hid_dim]
        return H0.unsqueeze(0)

    # @torchsnoop()
    def forward_decoder(self, inputs, hidden, cell, H0):
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, lstm hid dim]
        # cell = [n layers, batch size, lstm_hid dim]
        inputs = inputs.unsqueeze(0)
        # input = [seq_len = 1, batch size]
        embedded = self.dropout(self.embedding(inputs))
        embedded = torch.cat([embedded, H0], dim=2)
        # embedded = [1, batch size, emb dim + lstm_hid_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [1, batch size, lstm_hid dim]
        # hidden = [n layers = 1, batch size, lstm_hid dim]
        # cell = [n layers, batch size, lstm hid dim]

        prediction = self.fc_pred(output.squeeze(0))
        # prediction = [batch size, vocab size]

        return prediction, hidden, cell

    # @torchsnoop()
    def forward(self, src, trg, teacher_forcing_ratio=1):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, self.vocab_size).to(self.device)

        H0 = self.forward_encoder(src, trg)
        hidden = H0
        # first input to the decoder is the <sos> tokens
        inputs = trg[0, :]
        cell = torch.zeros_like(H0)
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.forward_decoder(inputs, hidden, cell, H0)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            inputs = trg[t] if teacher_force else top1

        return outputs


class GeneratorSeq2seq(nn.Module):
    def __init__(self, vocab_size, emb_dim, context_encoder,
                 lstm_input_dim, lstm_hid_dim, n_layers=1, dropout=0,
                 padding_idx=None, device='cpu'
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.context_encoder = context_encoder
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hid_dim, num_layers=n_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc_pred = nn.Linear(in_features=lstm_hid_dim, out_features=vocab_size, bias=False)
        self.encoder_decoder = nn.ModuleList([self.embedding, self.context_encoder,
                                              self.rnn, self.fc_pred]
                                             )

    # @torchsnoop()
    def forward_encoder(self, src, trg):
        src_embed = self.embedding(src.T)
        context = self.context_encoder(src_embed)
        # context = [1, batch_size, lstm_hid_dim]
        return context.unsqueeze(0)

    def forward_decoder(self, inputs, hidden, cell, H0):
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, lstm hid dim]
        # cell = [n layers, batch size, lstm_hid dim]
        inputs = inputs.unsqueeze(0)
        # input = [seq_len = 1, batch size]
        embedded = self.dropout(self.embedding(inputs))
        embedded = torch.cat([embedded, H0], dim=2)
        # embedded = [1, batch size, emb dim + lstm_hid_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [1, batch size, lstm_hid dim]
        # hidden = [n layers = 1, batch size, lstm_hid dim]
        # cell = [n layers, batch size, lstm hid dim]

        prediction = self.fc_pred(output.squeeze(0))
        # prediction = [batch size, vocab size]

        return prediction, hidden, cell

    def forward(self, src, trg, teacher_forcing_ratio=1):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, self.vocab_size).to(self.device)

        H0 = self.forward_encoder(src, trg)
        hidden = H0
        # first input to the decoder is the <sos> tokens
        inputs = trg[0, :]
        cell = torch.zeros_like(H0)
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.forward_decoder(inputs, hidden, cell, H0)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            inputs = trg[t] if teacher_force else top1

        return outputs


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu())
        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        # outputs is now a non-packed sequence, all hidden states obtained
        # when the input is a pad token are all zeros
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer
        # assume 1 direction
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, cell):
        # inputs = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        inputs = inputs.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(inputs))
        # embedded = [1, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def create_mask(self, src):
        mask = (src != self.src_pad_idx)
        # mask.T = [batch size, src len]
        return mask.T

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        inputs = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(inputs, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            inputs = trg[t] if teacher_force else top1

        return outputs
