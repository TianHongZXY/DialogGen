from seq2seq import Generator, ConvEncoder, Discriminator, Classifier2layer
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
from pysnooper import snoop
from torchsnooper import snoop as torchsnoop
from itertools import chain
import os


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
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--fea_dim', default=100, type=int, help='feature dim')
    parser.add_argument('--n_filters', default=300, type=int, help='filters num of conv encoder')
    parser.add_argument('--emb_dim', default=300, type=int, help='embedding dim')
    parser.add_argument('--kernel_size', default=5, type=int, help='kernel size of conv encoder')
    parser.add_argument('--stride', default=2, type=int, help='stride of conv encoder')
    parser.add_argument('--hid_dim', default=500, type=int, help='hidden dim of lstm')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout ratio')
    parser.add_argument('--n_epochs', default=30, type=int, help='num of train epoch')
    parser.add_argument('--clip', default=None, type=float, help='grad clip')
    parser.add_argument('--lamda', default=1e-1, type=float, help='weight of loss_decorr')
    parser.add_argument('--teach', default=1, type=float, help='propability of using teacher')
    parser.add_argument('--maxlen', default=29, type=int, help='fixed length of text')
    parser.add_argument('--train_file', default=None, type=str, help='train file path')
    parser.add_argument('--valid_file', default=None, type=str, help='valid file path')
    parser.add_argument('--test_file', default=None, type=str, help='test file path')
    parser.add_argument('--save_dir', default='.', type=str, help='save dir')
    args = parser.parse_args()
    device = torch.device(args.gpu if (torch.cuda.is_available() and args.gpu >= 0) else 'cpu')
    batch_size = args.bs
    feature_dim = args.fea_dim
    n_filters = args.n_filters
    embed_dim = args.emb_dim
    kernel_size = args.kernel_size
    stride = args.stride
    agg_output_dim = n_filters
    lstm_hid_dim = args.hid_dim
    lstm_input_dim = embed_dim + lstm_hid_dim
    n_layers = 1
    K = 1
    dropout = args.dropout
    n_epochs = args.n_epochs
    grad_clip = args.clip
    lamda = args.lamda
    teacher_forcing_ratio = args.teach
    save_dir = os.path.join(args.save_dir, 'run' + str(int(time.time())))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sent_len1 = maxlen = 29  # dailydialog数据集平均每条src+trg有29个单词
    sent_len2 = math.floor((sent_len1 - kernel_size) / stride) + 1
    sent_len3 = math.floor((sent_len2 - kernel_size) / stride) + 1
    train_file_path = args.train_file
    val_file_path = args.valid_file
    test_file_path = args.test_file

    dataset = load_data(train_file_path=train_file_path, val_file_path=val_file_path,
                        test_file_path=test_file_path, batch_size=batch_size, device=device, maxlen=maxlen)
    texts = dataset['fields'][0]
    vocab_size = len(texts.vocab)
    PAD_IDX = texts.vocab.stoi[texts.pad_token]
    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    best_valid_loss = float('inf')
    best_bleu = 0
    discriminator = Discriminator(vocab_size=vocab_size, emb_dim=embed_dim, padding_idx=PAD_IDX,
                                  feature_dim=feature_dim, n_filters=n_filters, kernel_size=kernel_size,
                                  stride=stride, dropout=dropout, sent_len3=sent_len3).to(device)
    disc_embedding = discriminator.embedding
    t_encoder = discriminator.topic_encoder
    t_classifier = Classifier2layer(input_dim=2 * feature_dim, num_class=2).to(device)
    t_optimizer = optim.Adam(chain(disc_embedding.parameters(), t_encoder.parameters(), t_classifier.parameters()), lr=1e-4)
    p_encoder = discriminator.persona_encoder
    p_classifier = Classifier2layer(input_dim=2 * feature_dim, num_class=2).to(device)
    p_optimizer = optim.Adam(chain(disc_embedding.parameters(), p_encoder.parameters(), p_classifier.parameters()), lr=1e-4)

    if args.model == 'disc':
        # TODO 完善加载训练过的模型继续训练
        discriminator.apply(init_weights)
        print(f'The model has {count_parameters(discriminator) + count_parameters(p_classifier) + count_parameters(t_classifier):,} trainable parameters')
        for epoch in range(n_epochs):
            start_time = time.time()
            train_metrics = train_discriminator(model=discriminator, epoch=epoch, t_classifier=t_classifier,
                                                p_classifier=p_classifier, p_optimizer=p_optimizer,
                                                iterator=dataset['train_iterator'], t_optimizer=t_optimizer,
                                                criterion=bce_criterion, clip=grad_clip, lamda=lamda
                                                )
            valid_metrics = evaluate_discriminator(model=discriminator, epoch=epoch, t_classifier=t_classifier,
                                                   p_classifier=p_classifier,
                                                   iterator=dataset['valid_iterator'], criterion=bce_criterion, lamda=lamda)
            test_metrics = evaluate_discriminator(model=discriminator, epoch=epoch, t_classifier=t_classifier,
                                                  p_classifier=p_classifier,
                                                  iterator=dataset['test_iterator'], criterion=bce_criterion, lamda=lamda)
            # TODO 完善print_metrics
            train_loss_xent = train_metrics['epoch_loss_xent']
            train_loss_decorr = train_metrics['epoch_loss_DeCorr']
            valid_loss_xent = valid_metrics['epoch_loss_xent']
            valid_loss_decorr = valid_metrics['epoch_loss_DeCorr']
            valid_topic_acc = valid_metrics['epoch_topic_acc']
            valid_person_acc = valid_metrics['epoch_person_acc']
            valid_loss = valid_loss_xent + valid_loss_decorr
            test_loss_xent = test_metrics['epoch_loss_xent']
            test_loss_decorr = test_metrics['epoch_loss_DeCorr']
            test_topic_acc = test_metrics['epoch_topic_acc']
            test_person_acc = test_metrics['epoch_person_acc']
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if args.save and valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(discriminator.state_dict(), os.path.join(save_dir, f'best-disc-{epoch}.pt'))

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss xent: {train_loss_xent:.3f} | Train Loss DeCorr: {train_loss_decorr:.3f} | ')
            print(f'\tVal Loss xent: {valid_loss_xent:.3f} | Val Loss DeCorr: {valid_loss_decorr:.3f} | '
                  f'Val topic acc: {valid_topic_acc:.3f} | Val person acc: {valid_person_acc:.3f} | ')

            print(f'\tTest Loss xent: {test_loss_xent:.3f} | Test Loss DeCorr: {test_loss_decorr:.3f} | '
                  f'Test topic acc: {test_topic_acc:.3f} | Test person acc: {test_person_acc:.3f} | ')
    elif args.model == 'gen':
        discriminator.load_state_dict(torch.load('models/best-disc-model.pt'))
        for param in discriminator.parameters():
            param.requires_grad = False
        # 暂时不共享lstm和disc的embedding
        # discriminator.embedding.requires_grad = True
        context_encoder = ConvEncoder(emb_dim=embed_dim, hid_dim=n_filters, output_dim=feature_dim,
                                      kernel_size=kernel_size, stride=stride, dropout=dropout, sent_len3=sent_len3)
        model = Generator(vocab_size=vocab_size, emb_dim=embed_dim, discriminator=discriminator,
                          context_encoder=context_encoder, agg_output_dim=agg_output_dim, lstm_hid_dim=lstm_hid_dim,
                          lstm_input_dim=lstm_input_dim, n_layers=n_layers, dropout=dropout, K=K, device=device).to(device)

        print(f'The model has {count_parameters(model):,} trainable parameters')
        # TODO 完善加载模型继续训练
        model.encoder_decoder.apply(init_weights)
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-4)
        
        bleu = BLEU(exclude_indices={PAD_IDX, texts.vocab.stoi[texts.eos_token], texts.vocab.stoi[texts.init_token]})

        for epoch in range(n_epochs):
            start_time = time.time()
            train_metrics = train_generator(model, dataset['train_iterator'], optimizer, ce_criterion, grad_clip, teacher_forcing_ratio=teacher_forcing_ratio)
            valid_metrics = evaluate_generator(model, dataset['valid_iterator'], ce_criterion, bleu=bleu)
            train_loss = train_metrics['epoch_loss']
            valid_loss = valid_metrics['epoch_loss']
            valid_bleu = valid_metrics['bleu']
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # if best_bleu < valid_bleu:
            # if valid_loss < best_valid_loss:
            #     best_valid_loss = valid_loss
            #     print('best valid loss: {:.3f}'.format(best_valid_loss))
            #     print('best PPL: {:.3f}'.format(math.exp(best_valid_loss)))
            #     print('current bleu: ', best_bleu)
            #     torch.save(model.encoder_decoder.state_dict(), 'models/best-enc-dec-model.pt')
            #     torch.save(model, 'models/full_model.pt')
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\tVal. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU: {valid_bleu:.5f} |')
            if args.save:
                torch.save(model.encoder_decoder.state_dict(), os.path.join(save_dir, f'enc-dec-model-{epoch}.pt'))


# model.encoder_decoder.load_state_dict(torch.load('models/best-enc-dec-model.pt'))
        # test_metrics = evaluate_generator(model, dataset['test_iterator'], ce_criterion, bleu=bleu)
        # test_loss = test_metrics['epoch_loss']
        # test_bleu = test_metrics['bleu']
        # test_generator(model, dataset['test_iterator'], text_field=texts)
        #
        # print(f'\tTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU: {test_bleu:.5f} |')


def train_generator(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio=1):
    model.train()

    epoch_loss_mle = 0
    # i = 0
    for batch in tqdm(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        loss_mle = criterion(output, trg)

        loss_mle.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss_mle += loss_mle.item()
        # i += 1
        # if i == 10:
        # break
    return {'epoch_loss': epoch_loss_mle / len(iterator)}


def evaluate_generator(model, iterator, criterion, bleu=None):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for batch in tqdm(iterator):
            src = batch.src
            # trg = [trg len, batch size]
            trg = batch.trg

            # output = [trg len, batch size, output dim]
            output = model(src, trg, 0)  # turn off teacher forcing
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


def disentangling_loss(feature):
    # feature = [batch_size, hid_dim]
    M = corrcoef(feature, rowvar=False)
    # M = [hid_dim, hid_dim]
    loss_decorr = 0.5 * (torch.sum(torch.pow(M, 2)) - torch.sum(torch.pow(torch.diag(M), 2)))
    return loss_decorr


# @snoop()
# @torchsnoop()
def train_discriminator(model, p_classifier, t_classifier, p_optimizer, t_optimizer,
                        iterator, criterion, clip, epoch, lamda=1e-2, reg=1e-3):
    model.train()
    p_classifier.train()
    t_classifier.train()
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
            p_optimizer.zero_grad()
            t_optimizer.zero_grad()
            # t_hat = [batch_size, ]
            feature_dict = model(src, trg)
            src_t_feature = F.sigmoid(feature_dict['src_t_feature'])
            trg_t_feature = F.sigmoid(feature_dict['trg_t_feature'])
            # t_hat = F.sigmoid(torch.bmm(src_t_feature.unsqueeze(1), trg_t_feature.unsqueeze(1).permute(0, 2, 1))).squeeze()
            t_hat = torch.mean(src_t_feature * trg_t_feature - 0.5, dim=1)
            # t_hat = t_classifier(torch.cat([src_t_feature, trg_t_feature], dim=1))
            src_p_feature = F.sigmoid(feature_dict['src_p_feature'])
            trg_p_feature = F.sigmoid(feature_dict['trg_p_feature'])
            # p_hat = F.sigmoid(torch.bmm(src_p_feature.unsqueeze(1), trg_p_feature.unsqueeze(1).permute(0, 2, 1))).squeeze()
            p_hat = torch.mean(src_p_feature * trg_p_feature - 0.5, dim=1)
            # p_hat = p_classifier(torch.cat([src_p_feature, trg_p_feature], dim=1))
            loss_t = criterion(t_hat, same_topic)
            loss_p = criterion(p_hat, same_person)
            # 作者开源代码里加的loss，说是鼓励binary
            # loss_binary_t = reg * torch.mean(torch.square(torch.ones_like(src_t_feature)-src_t_feature) * torch.square(src_t_feature)) \
            #               + reg * torch.mean(torch.square(torch.ones_like(trg_t_feature)-trg_t_feature) * torch.square(trg_t_feature))
            # loss_binary_p = reg * torch.mean(torch.square(torch.ones_like(src_p_feature)-src_p_feature) * torch.square(src_p_feature)) \
            #               + reg * torch.mean(torch.square(torch.ones_like(trg_p_feature)-trg_p_feature) * torch.square(trg_p_feature))
            loss_xent = loss_t + loss_p
            loss_DeCorr = disentangling_loss(src_t_feature) + disentangling_loss(trg_p_feature) + \
                          disentangling_loss(src_p_feature) + disentangling_loss(trg_t_feature)
            loss_D = loss_xent + lamda * loss_DeCorr
            loss_D.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(chain(model.parameters(), t_classifier.parameters(), p_classifier.parameters()), clip)
            p_optimizer.step()
            t_optimizer.step()

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


def evaluate_discriminator(model, p_classifier, t_classifier, iterator, criterion, epoch, lamda=1e-2):
    model.eval()
    p_classifier.eval()
    t_classifier.eval()
    passed_batch = 0
    epoch_loss_xent = 0
    epoch_loss_DeCorr = 0
    num_batches = len(iterator)
    batch_size = iterator.batch_size
    t_correct = 0
    p_correct = 0
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
                src_t_feature = F.sigmoid(feature_dict['src_t_feature'])
                trg_t_feature = F.sigmoid(feature_dict['trg_t_feature'])
                t_hat = torch.mean(src_t_feature * trg_t_feature - 0.5, dim=1)
                # t_hat = t_classifier(torch.cat([src_t_feature, trg_t_feature], dim=1))
                # t_hat = F.sigmoid(torch.bmm(src_t_feature.unsqueeze(1), trg_t_feature.unsqueeze(1).permute(0, 2, 1))).squeeze()
                src_p_feature = F.sigmoid(feature_dict['src_p_feature'])
                trg_p_feature = F.sigmoid(feature_dict['trg_p_feature'])
                p_hat = torch.mean(src_p_feature * trg_p_feature - 0.5, dim=1)
                # p_hat = p_classifier(torch.cat([src_p_feature, trg_p_feature], dim=1))
                # p_hat = F.sigmoid(torch.bmm(src_p_feature.unsqueeze(1), trg_p_feature.unsqueeze(1).permute(0, 2, 1))).squeeze()
                loss_t = criterion(t_hat, same_topic)
                loss_p = criterion(p_hat, same_person)
                loss_xent = loss_t + loss_p
                loss_DeCorr = disentangling_loss(src_t_feature) + disentangling_loss(trg_p_feature) + \
                              disentangling_loss(src_p_feature) + disentangling_loss(trg_t_feature)
                # t_pred = t_hat.argmax(1)
                # p_pred = p_hat.argmax(1)
                t_pred = (t_hat >= 0.5).int()
                p_pred = (p_hat >= 0.5).int()
                t_correct += torch.sum((t_pred == same_topic).int())
                p_correct += torch.sum((p_pred == same_person).int())
                epoch_loss_xent += loss_xent.item()
                epoch_loss_DeCorr += loss_DeCorr.item()
                t.set_postfix(batch_loss_xent=str(float('%.3f' % loss_xent.item())),
                              batch_loss_DeCorr=str(float('%.3f' % loss_DeCorr.item())))
                t.update(1)

    return {'epoch_loss_xent': epoch_loss_xent / (len(iterator) - passed_batch),
            'epoch_loss_DeCorr': epoch_loss_DeCorr / (len(iterator) - passed_batch),
            'epoch_topic_acc': t_correct / ((num_batches - passed_batch) * batch_size),
            'epoch_person_acc': p_correct / ((num_batches - passed_batch) * batch_size),
            }


if __name__ == '__main__':
    main()
