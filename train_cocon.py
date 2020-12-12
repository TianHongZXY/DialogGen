from seq2seq import Generator, ConvEncoder, Discriminator
from prepare_data import load_cocon_data as load_data
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
import math
import numpy as np
from tqdm import tqdm
from allennlp.training.metrics import BLEU
from train_seq2seq import init_weights, count_parameters, epoch_time
import argparse
from time import sleep

def print_metrics():
    # TODO 把print metrics统一起来
    pass


def main():
    parser = argparse.ArgumentParser(
        description='CoCon'
    )

    parser.add_argument('--gpu', default=-1, type=int, help='which GPU to use, -1 means using CPU')
    parser.add_argument('--save', default=False, type=bool, help='whether to save model or not')
    parser.add_argument('--model', default='disc', type=str, help='choose to train which model: {`disc`, `gen`}, (default: `disc`)')

    args = parser.parse_args()
    device = torch.device(args.gpu if (torch.cuda.is_available() and args.gpu >= 0) else 'cpu')
    batch_size = 128
    feature_dim = 100
    n_filters = 300
    embed_dim = 300
    kernel_size = 5
    stride = 2
    agg_output_dim = n_filters
    lstm_hid_dim = 500
    lstm_input_dim = embed_dim + lstm_hid_dim
    n_layers = 1
    K = 1
    dropout = 0.5
    n_epochs = 10
    grad_clip = 5
    lamda = 1e-2
    dataset = load_data(batch_size=batch_size, device=device)
    texts = dataset['fields'][0]
    vocab_size = len(texts.vocab)
    PAD_IDX = texts.vocab.stoi[texts.pad_token]
    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()

    best_valid_loss = float('inf')
    discriminator = Discriminator(vocab_size=vocab_size, emb_dim=embed_dim, padding_idx=PAD_IDX,
                                  feature_dim=feature_dim, n_filters=n_filters, kernel_size=kernel_size,
                                  stride=stride).to(device)

    if args.model == 'disc':
        # TODO 如果加载模型，这步应该省略
        discriminator.apply(init_weights)
        print(f'The model has {count_parameters(discriminator):,} trainable parameters')
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

        for epoch in range(n_epochs):
            start_time = time.time()
            train_metrics = train_discriminator(model=discriminator, epoch=epoch,
                                                iterator=dataset['train_iterator'], optimizer=disc_optimizer,
                                                criterion=bce_criterion, clip=grad_clip, lamda=lamda
                                                )
            # TODO 完善eval代码
            valid_metrics = evaluate_discriminator(model=discriminator, epoch=epoch,
                                                   iterator=dataset['valid_iterator'], criterion=bce_criterion, lamda=lamda)
            test_metrics = evaluate_discriminator(model=discriminator, epoch=epoch,
                                                  iterator=dataset['test_iterator'], criterion=bce_criterion, lamda=lamda)
            train_loss_xent = train_metrics['epoch_loss_xent']
            train_loss_decorr = train_metrics['epoch_loss_DeCorr']
            valid_loss_xent = valid_metrics['epoch_loss_xent']
            valid_loss_decorr = valid_metrics['epoch_loss_DeCorr']
            valid_loss = valid_loss_xent + valid_loss_decorr
            test_loss_xent = test_metrics['epoch_loss_xent']
            test_loss_decorr = test_metrics['epoch_loss_DeCorr']
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(discriminator.state_dict(), 'models/best-disc-model.pt')

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss xent: {train_loss_xent:.3f} | Train Loss DeCorr: {train_loss_decorr:.3f} | ')
            print(f'\tVal Loss xent: {valid_loss_xent:.3f} | Val Loss DeCorr: {valid_loss_decorr:.3f} | ')
            print(f'\tTest Loss xent: {test_loss_xent:.3f} | Test Loss DeCorr: {test_loss_decorr:.3f} | ')
    elif args.model == 'gen':
        discriminator.load_state_dict(torch.load('models/best-disc-model.pt'))
        for param in discriminator.parameters():
            param.requires_grad = False
        context_encoder = ConvEncoder(emb_dim=embed_dim, hid_dim=n_filters, output_dim=feature_dim,
                                      kernel_size=kernel_size, stride=stride)
        model = Generator(vocab_size=vocab_size, emb_dim=embed_dim, discriminator=discriminator,
                          context_encoder=context_encoder, agg_output_dim=agg_output_dim, lstm_hid_dim=lstm_hid_dim,
                          lstm_input_dim=lstm_input_dim, n_layers=n_layers, dropout=dropout, K=K, device=device).to(device)

        print(f'The model has {count_parameters(model):,} trainable parameters')
        # TODO 如果加载模型，这步应该省略
        model.encoder_decoder.apply(init_weights)
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-4)
        
        bleu = BLEU(exclude_indices={PAD_IDX, SRC.vocab.stoi[SRC.eos_token], SRC.vocab.stoi[SRC.init_token]})

        for epoch in range(n_epochs):
            start_time = time.time()
            train_metrics = train(model, dataset['train_iterator'], optimizer, ce_criterion, grad_clip)
            valid_metrics = evaluate(model, dataset['valid_iterator'], ce_criterion, bleu=bleu)
            train_loss = train_metrics['epoch_loss']
            valid_loss = valid_metrics['epoch_loss']
            valid_bleu = valid_metrics['bleu']
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.encoder_decoder.state_dict(), 'best-enc-dec-model.pt')

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\tVal. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU: {valid_bleu:.5f} |')

        model.encoder_decoder.load_state_dict(torch.load('best-gen-model.pt'))
        test_metrics = evaluate(model, dataset['test_iterator'], ce_criterion, bleu=bleu)
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

from pysnooper import snoop

# @snoop()
def corrcoef(x, rowvar=True):
    """
    code from
    https://github.com/AllenCellModeling/pytorch_integrated_cell/blob/8a83fc6f8dc79037f4b681d9d7ef0bc5b91e9948/integrated_cell/corr_stats.py
    Mimics `np.corrcoef`
    Arguments
    ---------
    x : 2D torch.Tensor
    rowvar : bool, default True means every single row is a variable, and every single column is an observation, e.g. a sample
    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)
    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref:
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013
    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    # 计算每个变量的均值，默认每行是一个变量，每列是一个sample
    if not rowvar and len(x.size()) != 1:
        x = x.T
    mean_x = torch.mean(x, 1).unsqueeze(1)
    # xm(j, i)是第i个sample的第j个变量，已经被减去了j变量的均值，等于论文中的F(si)j- uj,
    # xm(k, i)是第i个sample的第k个变量，已经被减去了k变量的均值，等于论文中的F(si)k- uk,
    xm = x.sub(mean_x.expand_as(x))
    # c(j, k) 等于论文中 M(j, k)的分子, c也是F(s)的协方差矩阵Cov(F(s), F(s))
    c = xm.mm(xm.t())
    # 协方差矩阵一般会除以 num_sample - 1
    # c = c / (x.size(1) - 1)

    # normalize covariance matrix
    # dj是每个变量的方差, E[(F(s)j - uj)^2]，也即j == k 时的分子
    d = torch.diag(c)
    # 取标准差
    stddev = torch.pow(d + 1e-7, 0.5)  # 防止出现0，导致nan
    # 论文中除以的分母
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c


# @snoop()
def disentangling_loss(feature):
    # feature = [batch_size, hid_dim]
    M = corrcoef(feature, rowvar=False)
    # M = [hid_dim, hid_dim]
    loss_decorr = 0.5 * (torch.sum(torch.pow(M, 2)) - torch.sum(torch.pow(torch.diag(M), 2)))
    return loss_decorr


def train_discriminator(model, iterator, optimizer, criterion, clip, epoch, lamda=1e-2):
    model.train()
    passed_batch = 0
    epoch_loss_xent = 0
    epoch_loss_DeCorr = 0
    with tqdm(total=len(iterator)) as t:
        for batch in iterator:
            t.set_description('Epoch %i' % epoch)
            # src = [src len, batch_size].T
            src = batch.src.T
            trg = batch.trg.T
            if src.size(1) < model.kernel_size or trg.size(1) < model.kernel_size:
                passed_batch += 1
                continue
            # same_{} = [batch_size, ]
            same_person = batch.same_person.squeeze(-1).float()
            same_topic = batch.same_topic.squeeze(-1).float()
            optimizer.zero_grad()
            # t_hat = [batch_size, ]
            feature_dict = model(src, trg)
            src_t_feature = feature_dict['src_t_feature']
            trg_t_feature = feature_dict['trg_t_feature']
            t_hat = F.sigmoid(torch.bmm(src_t_feature.unsqueeze(1), trg_t_feature.unsqueeze(1).permute(0, 2, 1))).squeeze()
            src_p_feature = feature_dict['src_p_feature']
            trg_p_feature = feature_dict['trg_p_feature']
            p_hat = F.sigmoid(torch.bmm(src_p_feature.unsqueeze(1), trg_p_feature.unsqueeze(1).permute(0, 2, 1))).squeeze()
            loss_t = criterion(t_hat, same_topic)
            loss_p = criterion(p_hat, same_person)
            loss_xent = loss_t + loss_p
            loss_DeCorr = disentangling_loss(src_t_feature) + disentangling_loss(trg_p_feature) + \
                          disentangling_loss(src_p_feature) + disentangling_loss(trg_t_feature)
            loss_D = loss_xent + lamda * loss_DeCorr
            loss_D.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss_xent += loss_xent.item()
            epoch_loss_DeCorr += loss_DeCorr.item()
            # tqdm.write('batch_loss_xent: ' + str(loss_xent.item()))
            t.set_postfix(batch_loss_xent=str(float('%.3f' % loss_xent.item())),
                          batch_loss_DeCorr=str(float('%.3f' % loss_DeCorr.item())))
            # tqdm.write('batch_loss_DeCorr:' + str(loss_DeCorr.item()))
            # print('batch_loss_xent', loss_xent.item())
            # print('batch_loss_DeCorr', loss_DeCorr.item())
            t.update(1)

    return {'epoch_loss_xent': epoch_loss_xent / (len(iterator) - passed_batch),
            'epoch_loss_DeCorr': epoch_loss_DeCorr / (len(iterator) - passed_batch)}


def evaluate_discriminator(model, iterator, criterion, epoch, lamda=1e-2):
    model.eval()
    passed_batch = 0
    epoch_loss_xent = 0
    epoch_loss_DeCorr = 0
    with torch.no_grad():
        with tqdm(total=len(iterator)) as t:
            for batch in iterator:
                t.set_description('Epoch %i' % epoch)
                # src = [src len, batch_size].T
                src = batch.src.T
                trg = batch.trg.T
                if src.size(1) < model.kernel_size or trg.size(1) < model.kernel_size:
                    passed_batch += 1
                    continue
                # same_{} = [batch_size, ]
                same_person = batch.same_person.squeeze(-1).float()
                same_topic = batch.same_topic.squeeze(-1).float()
                # t_hat = [batch_size, ]
                feature_dict = model(src, trg)
                src_t_feature = feature_dict['src_t_feature']
                trg_t_feature = feature_dict['trg_t_feature']
                t_hat = F.sigmoid(torch.bmm(src_t_feature.unsqueeze(1), trg_t_feature.unsqueeze(1).permute(0, 2, 1))).squeeze()
                src_p_feature = feature_dict['src_p_feature']
                trg_p_feature = feature_dict['trg_p_feature']
                p_hat = F.sigmoid(torch.bmm(src_p_feature.unsqueeze(1), trg_p_feature.unsqueeze(1).permute(0, 2, 1))).squeeze()
                loss_t = criterion(t_hat, same_topic)
                loss_p = criterion(p_hat, same_person)
                loss_xent = loss_t + loss_p
                loss_DeCorr = disentangling_loss(src_t_feature) + disentangling_loss(trg_p_feature) + \
                              disentangling_loss(src_p_feature) + disentangling_loss(trg_t_feature)

                epoch_loss_xent += loss_xent.item()
                epoch_loss_DeCorr += loss_DeCorr.item()
                t.set_postfix(batch_loss_xent=str(float('%.3f' % loss_xent.item())),
                              batch_loss_DeCorr=str(float('%.3f' % loss_DeCorr.item())))
                t.update(1)

    return {'epoch_loss_xent': epoch_loss_xent / (len(iterator) - passed_batch),
            'epoch_loss_DeCorr': epoch_loss_DeCorr / (len(iterator) - passed_batch)}
if __name__ == '__main__':
    main()
