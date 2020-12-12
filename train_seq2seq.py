from seq2seq import Seq2Seq, Encoder, Decoder
from prepare_data import load_data
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
import math
import numpy as np
from tqdm import tqdm
from allennlp.training.metrics import BLEU


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def init_weights(m):
#     for name, param in m.named_parameters():
#         nn.init.uniform_(param.data, -0.08, 0.08)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # nn.init.xavier_uniform_(module.weight.data)
        nn.init.orthogonal(module.weight)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if module.bidirectional:
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0

    elif isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight.data)

    else:
        for param in module.parameters():
            nn.init.normal_(param)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    dataset = load_data(batch_size=32, device=device)
    SRC = dataset['fields'][0]
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(SRC.vocab)
    ENC_EMB_DIM = 100
    DEC_EMB_DIM = 100
    HID_DIM = 100
    N_LAYERS = 1
    ENC_DROPOUT = 0
    DEC_DROPOUT = 0

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

    model = Seq2Seq(enc, dec, PAD_IDX, device).to(device)
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    N_EPOCHS = 10
    CLIP = 5

    best_valid_loss = float('inf')
    bleu = BLEU(exclude_indices={PAD_IDX, SRC.vocab.stoi[SRC.eos_token], SRC.vocab.stoi[SRC.init_token]})
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_metrics = train(model, dataset['train_iterator'], optimizer, criterion, CLIP)
        valid_metrics = evaluate(model, dataset['valid_iterator'], criterion, bleu=bleu)
        train_loss = train_metrics['epoch_loss']
        valid_loss = valid_metrics['epoch_loss']
        valid_bleu = valid_metrics['bleu']
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU: {valid_bleu:.5f} |')

    model.load_state_dict(torch.load('best-model.pt'))
    test_metrics = evaluate(model, dataset['test_iterator'], criterion, bleu=bleu)
    test_loss = test_metrics['epoch_loss']
    test_bleu = test_metrics['bleu']
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU: {test_bleu:.5f} |')


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    # i = 0
    for batch in tqdm(iterator):
        src, src_len = batch.src
        trg, trg_len = batch.trg

        optimizer.zero_grad()

        output = model(src, src_len, trg, teacher_forcing_ratio=1)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        # i += 1
        # if i == 10:
            # break
    return {'epoch_loss': epoch_loss / len(iterator)}


def evaluate(model, iterator, criterion, bleu=None):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for batch in tqdm(iterator):
            src, src_len = batch.src
            # trg = [trg len, batch size]
            trg, trg_len = batch.trg

            # output = [trg len, batch size, output dim]
            output = model(src, src_len, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]

            if bleu:
                gold = trg.T
                pred = output.argmax(-1).T
                bleu(predictions=pred, gold_targets=gold)

            # output = [(trg len - 1) * batch size, output dim]
            output = output[1:].reshape(-1, output_dim)
            # trg = [(trg len - 1) * batch size]
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return {'epoch_loss': epoch_loss / len(iterator), 'bleu': bleu.get_metric(reset=True)['BLEU']}


if __name__ == '__main__':
    main()
