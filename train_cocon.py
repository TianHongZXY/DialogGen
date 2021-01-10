from seq2seq import Generator, ConvEncoder, Discriminator, Classifier2layer, GeneratorSeq2seq
from prepare_data import load_cocon_data as load_data
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
from dateutil import tz, zoneinfo
import math
from tqdm import tqdm
from allennlp.training.metrics import BLEU
import argparse
from pysnooper import snoop
# from torchsnooper import snoop as torchsnoop
from itertools import chain
import os
from time import sleep
import json
from metrics import distinct
from utils import disentangling_loss, init_weights, count_parameters, epoch_time, print_metrics, write_metrics
from test_cocon import test_generator as evaluate_generator
import logging
import coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


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
    parser.add_argument('--save_dir', default='models', type=str, help='save dir')
    parser.add_argument('--pretrained_embed_file', default=None, type=str, help='glove file path')
    parser.add_argument('--trained_disc', default=None, type=str, help='load trained disc for training generator')
    parser.add_argument('--num_workers', default=0, type=int, help='how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('--vocab_file', default=None, type=str, help='vocab file path')
    parser.add_argument('--l2', default=0, type=float, help='l2 regularization')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--generator', default='cocon', type=str, help='which generator to use')
    # args = parser.parse_args()
    args, unparsed = parser.parse_known_args()
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

    sent_len1 = maxlen = 29  # dailydialog数据集平均每条src+trg有29个单词
    sent_len2 = math.floor((sent_len1 - kernel_size) / stride) + 1
    sent_len3 = math.floor((sent_len2 - kernel_size) / stride) + 1
    args.sent_len3 = sent_len3
    train_file_path = args.train_file
    val_file_path = args.valid_file
    test_file_path = args.test_file

    if args.save:
        tz_sh = tz.gettz('Asia/Shanghai')
        save_dir = os.path.join(args.save_dir, 'run' + str(datetime.now(tz=tz_sh)).replace(":", "-").split(".")[0].replace(" ", '.'))
        args.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        # with open(os.path.join(save_dir, 'vocab.txt'), 'w') as f:
        #     json.dump(texts.vocab.stoi, f)

    dataset = load_data(args, train_file_path=train_file_path, val_file_path=val_file_path,
                        test_file_path=test_file_path, batch_size=batch_size, device=device, maxlen=maxlen,
                        pretrained_embed_file=args.pretrained_embed_file)
    texts = dataset['fields'][0]
    vocab_size = len(texts.vocab)
    args.vocab_size = vocab_size

    PAD_IDX = texts.vocab.stoi[texts.pad_token]
    ce_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    bce_criterion = nn.BCEWithLogitsLoss()

    best_valid_loss = float('inf')
    best_valid_acc = -float('inf')
    best_epoch = -1
    best_bleu = 0
    if args.model == 'disc':
        discriminator = Discriminator(vocab_size=vocab_size, emb_dim=embed_dim, padding_idx=PAD_IDX,
                                      feature_dim=feature_dim, n_filters=n_filters, kernel_size=kernel_size,
                                      stride=stride, dropout=dropout, sent_len3=sent_len3).to(device)
        discriminator.apply(init_weights)
        disc_embedding = discriminator.embedding
        if texts.vocab.vectors is not None:
            discriminator.embedding.weight.data.copy_(texts.vocab.vectors)
        t_encoder = discriminator.topic_encoder
        p_encoder = discriminator.persona_encoder
        t_classifier = Classifier2layer(input_dim=2 * feature_dim, num_class=2).to(device)
        p_classifier = Classifier2layer(input_dim=2 * feature_dim, num_class=2).to(device)
        t_optimizer = optim.Adam(chain(disc_embedding.parameters(), t_encoder.parameters(), t_classifier.parameters()),
                                 lr=args.lr, weight_decay=args.l2)
        p_optimizer = optim.Adam(chain(disc_embedding.parameters(), p_encoder.parameters(), p_classifier.parameters()),
                                 lr=args.lr, weight_decay=args.l2)
        # TODO 完善加载训练过的模型继续训练
        print(f'The model has {count_parameters(discriminator) + count_parameters(p_classifier) + count_parameters(t_classifier):,} trainable parameters')
        for epoch in range(n_epochs):
            start_time = time.time()
            train_metrics = train_discriminator(model=discriminator, epoch=epoch, t_classifier=t_classifier,
                                                p_classifier=p_classifier, p_optimizer=p_optimizer,
                                                iterator=dataset['train_iterator'], t_optimizer=t_optimizer,
                                                criterion=ce_criterion, clip=grad_clip, lamda=lamda
                                                )
            valid_metrics = evaluate_discriminator(model=discriminator, epoch=epoch, t_classifier=t_classifier,
                                                   p_classifier=p_classifier,
                                                   iterator=dataset['valid_iterator'], criterion=ce_criterion, lamda=lamda)
            test_metrics = evaluate_discriminator(model=discriminator, epoch=epoch, t_classifier=t_classifier,
                                                  p_classifier=p_classifier,
                                                  iterator=dataset['test_iterator'], criterion=ce_criterion, lamda=lamda)
            valid_loss_xent = valid_metrics['epoch_loss_xent']
            valid_loss_decorr = valid_metrics['epoch_loss_DeCorr']
            valid_topic_acc = valid_metrics['epoch_topic_acc']
            valid_person_acc = valid_metrics['epoch_person_acc']
            valid_loss = valid_loss_xent + valid_loss_decorr
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            # if args.save and valid_loss < best_valid_loss:
            #     best_valid_loss = valid_loss
            if 0.5 * (valid_person_acc + valid_topic_acc) > best_valid_acc:
                best_valid_acc = 0.5 * (valid_person_acc + valid_topic_acc)
                best_epoch = epoch
            if args.save:
                torch.save(discriminator.state_dict(), os.path.join(save_dir, f'best-disc-{epoch}.pt'))
                with open(os.path.join(save_dir, f'log_epoch{epoch}.txt'), 'w') as log_file:
                    log_file.write(f'Epoch: {epoch:02}\n')
                    write_metrics(train_metrics, log_file, mode='Train')
                    write_metrics(valid_metrics, log_file, mode='Valid')
                    write_metrics(test_metrics, log_file, mode='Test')
                    log_file.write(f'\tBest epoch: {best_epoch:02}\n')
                    log_file.write(f'\tBest Val acc: {best_valid_acc:.3f}\n')

            print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
            print_metrics(train_metrics, mode='Train')
            print_metrics(valid_metrics, mode='Valid')
            print_metrics(test_metrics, mode='Test')
    elif args.model == 'gen':
        if args.trained_disc is None:
            raise ValueError('If train generator, a trained disc file must be given!')
        disc_save_dir = os.path.split(args.trained_disc)[0]
        trained_disc_args = argparse.Namespace()
        with open(os.path.join(disc_save_dir, 'args.txt'), 'r') as f:
            d = json.load(f)
            for key, value in d.items():
                setattr(trained_disc_args, key, value)
        discriminator = Discriminator(vocab_size=len(texts.vocab), emb_dim=trained_disc_args.emb_dim, padding_idx=PAD_IDX,
                                      feature_dim=trained_disc_args.fea_dim, n_filters=trained_disc_args.n_filters, kernel_size=trained_disc_args.kernel_size,
                                      stride=trained_disc_args.stride, dropout=trained_disc_args.dropout, sent_len3=trained_disc_args.sent_len3).to(device)
        discriminator.load_state_dict(torch.load(args.trained_disc, map_location={'cuda:0': 'cuda:' + str(args.gpu)}))
        for param in discriminator.parameters():
            param.requires_grad = False
        # 暂时不共享lstm和disc的embedding
        # discriminator.embedding.requires_grad = True
        context_encoder = ConvEncoder(emb_dim=embed_dim, hid_dim=n_filters, output_dim=feature_dim,
                                      kernel_size=kernel_size, stride=stride, dropout=dropout, sent_len3=sent_len3)
        if args.generator == 'cocon':
            model = Generator(vocab_size=vocab_size, emb_dim=embed_dim, discriminator=discriminator,
                              context_encoder=context_encoder, agg_output_dim=agg_output_dim, lstm_hid_dim=lstm_hid_dim,
                              lstm_input_dim=lstm_input_dim, n_layers=n_layers, dropout=dropout, K=K, device=device).to(device)
        elif args.generator =='seq2seq':
            model = GeneratorSeq2seq(vocab_size=vocab_size, emb_dim=embed_dim, context_encoder=context_encoder,
                                     lstm_hid_dim=lstm_hid_dim, lstm_input_dim=lstm_input_dim, n_layers=n_layers,
                                     dropout=dropout, device=device).to(device)
        else:
            raise ValueError(f'generator should be in [cocon, seq2seq], but get {args.generator}!')
        model.encoder_decoder.apply(init_weights)
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2)
        # TODO 完善加载模型继续训练
        if texts.vocab.vectors is not None:
            model.embedding.weight.data.copy_(texts.vocab.vectors)
        print(f'The model has {count_parameters(model):,} trainable parameters')

        bleu = BLEU(exclude_indices={PAD_IDX, texts.vocab.stoi[texts.eos_token], texts.vocab.stoi[texts.init_token]})

        for epoch in range(n_epochs):
            start_time = time.time()
            train_metrics = train_generator(model, dataset['train_iterator'], optimizer, ce_criterion, grad_clip, teacher_forcing_ratio=teacher_forcing_ratio)
            valid_metrics = evaluate_generator(model, iterator=dataset['valid_iterator'], criterion=ce_criterion, bleu=bleu, dist=distinct, text_field=texts)
            # test_metrics = evaluate_generator(model, iterator=dataset['test_iterator'], criterion=ce_criterion, bleu=bleu, dist=distinct, text_field=texts)
            valid_loss = valid_metrics['epoch_loss']
            valid_bleu = valid_metrics['bleu']
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
            print_metrics(train_metrics, mode='Train')
            print_metrics(valid_metrics, mode='Valid')
            # print_metrics(test_metrics, mode='Test')
            if best_bleu < valid_bleu:
                best_bleu = valid_bleu
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                print(f'Best epoch: {best_epoch:02}')
                print('best valid loss: {:.3f}'.format(best_valid_loss))
                print('best PPL: {:.3f}'.format(math.exp(best_valid_loss)))
            if args.save:
                torch.save(model.encoder_decoder.state_dict(), os.path.join(save_dir, f'enc-dec-model-{epoch}.pt'))
                with open(os.path.join(save_dir, f'log_epoch{epoch}.txt'), 'w') as log_file:
                    log_file.write(f'Epoch: {epoch:02}\n')
                    write_metrics(train_metrics, log_file, mode='Train')
                    write_metrics(valid_metrics, log_file, mode='Valid')
                    log_file.write(f'\tBest epoch: {best_epoch:02}\n')
                    log_file.write(f'\tBest Val Loss: {best_valid_loss:.3f}\n')
                    log_file.write(f'\tBest BLEU: {best_bleu:.7f}\n')


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
        # loss是在dim=0上平均后得到的，即除以了batch_size * trg_len，是每个词的cross entropy
        epoch_loss_mle += loss_mle.item()
        # i += 1
        # if i == 10:
        # break
    metrics = dict()
    metrics['epoch_loss'] = epoch_loss_mle / len(iterator)
    metrics['PPL'] = math.exp(metrics['epoch_loss'])
    return metrics


def old_evaluate_generator(model, iterator, criterion, bleu=None, dist=None):
    model.eval()

    epoch_loss = 0
    hyps = []
    # refs = []
    with torch.no_grad():
        for batch in tqdm(iterator):
            src = batch.src
            # trg = [trg len, batch size]
            trg = batch.trg

            # output = [trg len, batch size, output dim]
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]

            if bleu or dist:
                gold = trg.T[:, 1:-1]
                pred = output.argmax(-1).T[:, 1:-1]
                hyps.extend([h for h in pred])
                # refs.extend([r for r in gold])
                bleu(predictions=pred, gold_targets=gold)

            # output = [(trg len - 1) * batch size, output dim]
            output = output[1:].reshape(-1, output_dim)
            # trg = [(trg len - 1) * batch size]
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item() / src.size(1)
    metrics = dict()
    if dist:
        inter_dist1, inter_dist2 = dist(hyps)
        metrics['dist1'], metrics['dist2'] = inter_dist1, inter_dist2
    if bleu:
        metrics['bleu'] = bleu.get_metric(reset=True)['BLEU']
    metrics['epoch_loss'] = epoch_loss
    return metrics


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
        for i, batch in enumerate(iterator, start=1):
            t.set_description('Epoch %i' % epoch)
            # src = [src len, batch_size].T
            src = batch.src.T
            trg = batch.trg.T

            if isinstance(criterion, nn.CrossEntropyLoss):
                same_person = batch.same_person.squeeze(-1).long()
                same_topic = batch.same_topic.squeeze(-1).long()
            elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                same_person = batch.same_person.squeeze(-1).float()
                same_topic = batch.same_topic.squeeze(-1).float()
            p_optimizer.zero_grad()
            t_optimizer.zero_grad()
            # t_hat = [batch_size, ]
            feature_dict = model(src, trg)
            src_t_feature = feature_dict['src_t_feature']
            trg_t_feature = feature_dict['trg_t_feature']
            # t_hat = F.sigmoid(torch.bmm(src_t_feature.unsqueeze(1), trg_t_feature.unsqueeze(1).permute(0, 2, 1))).squeeze()
            # t_hat = torch.mean(src_t_feature * trg_t_feature - 0.5, dim=1)
            t_hat = t_classifier(torch.cat([src_t_feature, trg_t_feature], dim=1))
            src_p_feature = feature_dict['src_p_feature']
            trg_p_feature = feature_dict['trg_p_feature']
            # p_hat = F.sigmoid(torch.bmm(src_p_feature.unsqueeze(1), trg_p_feature.unsqueeze(1).permute(0, 2, 1))).squeeze()
            # p_hat = torch.mean(src_p_feature * trg_p_feature - 0.5, dim=1)
            p_hat = p_classifier(torch.cat([src_p_feature, trg_p_feature], dim=1))

            loss_t = criterion(t_hat, same_topic)
            loss_p = criterion(p_hat, same_person)
            # 作者开源代码里加的loss，说是鼓励binary
            # loss_binary_t = reg * torch.mean(torch.square(torch.ones_like(src_t_feature)-src_t_feature) * torch.square(src_t_feature)) \
            #               + reg * torch.mean(torch.square(torch.ones_like(trg_t_feature)-trg_t_feature) * torch.square(trg_t_feature))
            # loss_binary_p = reg * torch.mean(torch.square(torch.ones_like(src_p_feature)-src_p_feature) * torch.square(src_p_feature)) \
            #               + reg * torch.mean(torch.square(torch.ones_like(trg_p_feature)-trg_p_feature) * torch.square(trg_p_feature))
            loss_xent = loss_t + loss_p
            loss_DeCorr = torch.mean(disentangling_loss(src_t_feature) + disentangling_loss(trg_p_feature) + \
                          disentangling_loss(src_p_feature) + disentangling_loss(trg_t_feature))
            loss_D = loss_xent + lamda * loss_DeCorr
            loss_D.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(chain(model.parameters(), t_classifier.parameters(), p_classifier.parameters()), clip)
            p_optimizer.step()
            t_optimizer.step()

            epoch_loss_xent += loss_xent.item()
            epoch_loss_DeCorr += loss_DeCorr.item()
            t.set_postfix(batch_loss_xent=str(float('%.3f' % loss_xent.item())),
                          batch_loss_DeCorr=str(float('%.3f' % loss_DeCorr.item())))

            t.update(1)

    return {'epoch_loss_xent': epoch_loss_xent,
            'epoch_loss_DeCorr': epoch_loss_DeCorr}


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
                if isinstance(criterion, nn.CrossEntropyLoss):
                    same_person = batch.same_person.squeeze(-1).long()
                    same_topic = batch.same_topic.squeeze(-1).long()
                elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                    same_person = batch.same_person.squeeze(-1).float()
                    same_topic = batch.same_topic.squeeze(-1).float()
                # t_hat = [batch_size, ]
                feature_dict = model(src, trg)
                src_t_feature = feature_dict['src_t_feature']
                trg_t_feature = feature_dict['trg_t_feature']
                # t_hat = torch.mean(src_t_feature * trg_t_feature - 0.5, dim=1)
                t_hat = t_classifier(torch.cat([src_t_feature, trg_t_feature], dim=1))
                # t_hat = F.sigmoid(torch.bmm(src_t_feature.unsqueeze(1), trg_t_feature.unsqueeze(1).permute(0, 2, 1))).squeeze()
                src_p_feature = feature_dict['src_p_feature']
                trg_p_feature = feature_dict['trg_p_feature']
                # p_hat = torch.mean(src_p_feature * trg_p_feature - 0.5, dim=1)
                p_hat = p_classifier(torch.cat([src_p_feature, trg_p_feature], dim=1))
                # p_hat = F.sigmoid(torch.bmm(src_p_feature.unsqueeze(1), trg_p_feature.unsqueeze(1).permute(0, 2, 1))).squeeze()
                loss_t = criterion(t_hat, same_topic)
                loss_p = criterion(p_hat, same_person)
                loss_xent = loss_t + loss_p
                loss_DeCorr = torch.mean(disentangling_loss(src_t_feature) + disentangling_loss(trg_p_feature) + \
                              disentangling_loss(src_p_feature) + disentangling_loss(trg_t_feature))
                t_pred = t_hat.argmax(1)
                p_pred = p_hat.argmax(1)
                # t_pred = (t_hat >= 0.5).int()
                # p_pred = (p_hat >= 0.5).int()
                t_correct += torch.sum((t_pred == same_topic).int())
                p_correct += torch.sum((p_pred == same_person).int())
                epoch_loss_xent += loss_xent.item()
                epoch_loss_DeCorr += loss_DeCorr.item()
                t.set_postfix(batch_loss_xent=str(float('%.3f' % loss_xent.item())),
                              batch_loss_DeCorr=str(float('%.3f' % loss_DeCorr.item())))
                t.update(1)

    return {'epoch_loss_xent': epoch_loss_xent,
            'epoch_loss_DeCorr': epoch_loss_DeCorr,
            'epoch_topic_acc': t_correct / ((num_batches - passed_batch) * batch_size),
            'epoch_person_acc': p_correct / ((num_batches - passed_batch) * batch_size),
            }


if __name__ == '__main__':
    main()
