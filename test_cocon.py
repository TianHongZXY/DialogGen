from seq2seq import Generator, Discriminator, ConvEncoder
from prepare_data import load_cocon_data as load_data
import math
from allennlp.training.metrics import BLEU
from metrics import distinct
from tqdm import tqdm
import time
import torch
import argparse
import torch.nn.functional as F
import os
import json
from utils import print_metrics, disentangling_loss
import torch.nn as nn


def write_to_file(file, src_idx, gold_idx, pred_idx, vocab, pad_idx, eos_idx):
    batch_size = src_idx.size(0)
    for i in range(batch_size):
        src_str = ' '.join(list(map(lambda x: vocab.itos[x] if x != pad_idx and x != eos_idx else '', src_idx[i])))
        gold_str = ' '.join(list(map(lambda x: vocab.itos[x] if x != pad_idx and x != eos_idx else '', gold_idx[i])))
        pred_str = ' '.join(list(map(lambda x: vocab.itos[x] if x != pad_idx and x != eos_idx else '', pred_idx[i])))
        file.write('source: ' + src_str + '\n')
        file.write('gold: ' + gold_str + '\n')
        file.write('pred: ' + pred_str + '\n')

from pysnooper import snoop
from torchsnooper import snoop as torchsnooper
def test_generator(model, iterator, criterion, bleu=None, dist=None, file=None, text_field=None):
    model.eval()
    epoch_loss = 0
    hyps = []
    refs = []
    with torch.no_grad():
        for batch in tqdm(iterator):
            src = batch.src
            # trg = [trg len, batch size]
            trg = batch.trg
            # output = [trg len, batch size, output dim]
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            if bleu or dist:
                gold = trg.T[:, 1:]
                pred = output.argmax(-1).T[:, 1:]
                hyps.extend([h.cpu().numpy() for h in pred])
                refs.extend([r.cpu().numpy() for r in gold])
            if bleu is not None:
                bleu(predictions=pred, gold_targets=gold)
            if file is not None:
                write_to_file(file, src_idx=src.T[:, 1:], gold_idx=gold, pred_idx=pred,
                              vocab=text_field.vocab, pad_idx=text_field.vocab.stoi[text_field.pad_token],
                              eos_idx=text_field.vocab.stoi[text_field.eos_token])
            # output = [(trg len - 1) * batch size, output dim]
            output = output[1:].reshape(-1, output_dim)
            # trg = [(trg len - 1) * batch size]
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    metrics = dict()
    if dist is not None:
        inter_dist1, inter_dist2 = dist(hyps)
        metrics['dist1'], metrics['dist2'] = inter_dist1, inter_dist2
    if bleu is not None:
        metrics['bleu'] = bleu.get_metric(reset=True)['BLEU']
    metrics['epoch_loss'] = epoch_loss
    metrics['PPL'] = math.exp(metrics['epoch_loss'])
    return metrics

def old_test_generator(model, iterator, text_field, bleu=None, dist=None):
    model.eval()
    hyps = []
    refs = []
    with open('test_outputs.txt', 'w') as f:
        with torch.no_grad():
            for batch in tqdm(iterator):
                src = batch.src
                # trg = [trg len, batch size]
                trg = batch.trg

                # output = [trg len, batch size, output dim]
                output = model(src, trg, 0)  # turn off teacher forcing
                gold = trg.T[:, 1:-1]
                pred = output.argmax(-1).T[:, 1:-1]
                if bleu or dist:
                    gold = trg.T[:, 1:-1]
                    pred = output.argmax(-1).T[:, 1:-1]
                    hyps.extend([h for h in pred])
                    refs.extend([r for r in gold])
                write_to_file(f, src_idx=src.T[:, 1:-1], gold_idx=gold, pred_idx=pred,
                              vocab=text_field.vocab, pad_idx=text_field.vocab.stoi[text_field.pad_token])
                if bleu:
                    bleu(predictions=pred, gold_targets=gold)
    metrics = dict()
    if dist:
        inter_dist1, inter_dist2 = dist(hyps)
        metrics['dist1'], metrics['dist2'] = inter_dist1, inter_dist2
    if bleu:
        metrics['bleu'] = bleu.get_metric(reset=True)['BLEU']
    return metrics

def test_discriminator(model, iterator, p_classifier=None, t_classifier=None):
    model.eval()
    # p_classifier.eval()
    # t_classifier.eval()
    passed_batch = 0
    epoch_loss_DeCorr = 0
    num_batches = len(iterator)
    batch_size = iterator.batch_size
    t_correct = 0
    p_correct = 0
    with torch.no_grad():
        with tqdm(total=len(iterator)) as t:
            for batch in iterator:
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
                loss_DeCorr = disentangling_loss(src_t_feature) + disentangling_loss(trg_p_feature) + \
                              disentangling_loss(src_p_feature) + disentangling_loss(trg_t_feature)
                # t_pred = t_hat.argmax(1)
                # p_pred = p_hat.argmax(1)
                t_pred = (t_hat >= 0.5).int()
                p_pred = (p_hat >= 0.5).int()
                t_correct += torch.sum((t_pred == same_topic).int())
                p_correct += torch.sum((p_pred == same_person).int())
                epoch_loss_DeCorr += loss_DeCorr.item()
                # t.set_postfix(batch_loss_DeCorr=str(float('%.3f' % loss_DeCorr.item())))
                t.update(1)

    return {'epoch_loss_DeCorr': epoch_loss_DeCorr,
            'epoch_topic_acc': t_correct / ((num_batches - passed_batch) * batch_size),
            'epoch_person_acc': p_correct / ((num_batches - passed_batch) * batch_size),
            }

# def inference(context, text_field, model, device, max_len=50):
#     model.eval()
#
#     if isinstance(context, str):
#         tokens = text_field.tokenize(context)
#     elif isinstance(context, list):
#         tokens = [token.lower() for token in context]
#     else:
#         raise ValueError('context should be either str or list type!')
#     tokens = [text_field.init_token] + tokens + [text_field.eos_token]
#
#     context_indexes = [text_field.vocab.stoi[token] for token in tokens]
#
#     context_tensor = torch.LongTensor(context_indexes).unsqueeze(1).to(device)
#
#     context_len = torch.LongTensor([len(context_indexes)]).to(device)
#
#     with torch.no_grad():
#         encoder_outputs, hidden = model.forward_encoder(context_tensor, context_len)
#
#     trg_indexes = [text_field.vocab.stoi[text_field.init_token]]
#
#     attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
#
#     for i in range(max_len):
#
#         trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
#
#         with torch.no_grad():
#             output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
#
#         attentions[i] = attention
#
#         pred_token = output.argmax(1).item()
#
#         trg_indexes.append(pred_token)
#
#         if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
#             break
#
#     trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
#
#     return trg_tokens[1:], attentions[:len(trg_tokens) - 1]


def main():
    parser = argparse.ArgumentParser(
        description='CoCon'
    )

    parser.add_argument('--gpu', default=-1, type=int, help='which GPU to use, -1 means using CPU')
    parser.add_argument('--model', default='disc', type=str,
                        help='choose to test which model: {`disc`, `gen`}, (default: `disc`)')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--maxlen', default=29, type=int, help='fixed length of text')
    parser.add_argument('--test_file', default=None, type=str, help='test file path')
    parser.add_argument('--trained_disc', default=None, type=str, help='load trained discriminator')
    parser.add_argument('--trained_gen', default=None, type=str, help='load trained generator')
    parser.add_argument('--bleu', default=None, type=bool, help='True means compute, default False')
    parser.add_argument('--dist', default=None, type=bool, help='True means compute, default False')
    args, unparsed = parser.parse_known_args()
    if not args.trained_disc:
        raise ValueError('A trained disc file must be given!')
    disc_save_dir = os.path.split(args.trained_disc)[0]
    args.vocab_file = os.path.join(disc_save_dir, 'vocab.txt')
    device = torch.device(args.gpu if (torch.cuda.is_available() and args.gpu >= 0) else 'cpu')
    batch_size = args.bs
    n_layers = 1
    K = 1
    maxlen = 29  # dailydialog数据集平均每条src+trg有29个单词
    dataset = load_data(args, train_file_path=args.test_file, batch_size=batch_size, device=device,
                        maxlen=maxlen)
    texts = dataset['fields'][0]
    PAD_IDX = texts.vocab.stoi[texts.pad_token]
    disc_save_dir = os.path.split(args.trained_disc)[0]
    trained_disc_args = argparse.Namespace()
    with open(os.path.join(disc_save_dir, 'args.txt'), 'r') as f:
        d = json.load(f)
        for key, value in d.items():
            setattr(trained_disc_args, key, value)
    discriminator = Discriminator(vocab_size=len(texts.vocab), emb_dim=trained_disc_args.emb_dim,
                                  padding_idx=PAD_IDX,
                                  feature_dim=trained_disc_args.fea_dim, n_filters=trained_disc_args.n_filters,
                                  kernel_size=trained_disc_args.kernel_size,
                                  stride=trained_disc_args.stride, dropout=trained_disc_args.dropout,
                                  sent_len3=trained_disc_args.sent_len3).to(device)
    discriminator.load_state_dict(torch.load(args.trained_disc))
    if args.model == 'disc':
        test_metrics = test_discriminator(model=discriminator, iterator=dataset['test_iterator'])
        print_metrics(test_metrics)

    elif args.model == 'gen':
        if not args.trained_gen:
            raise ValueError('A trained gen file must be given if you want to test it!')
        gen_save_dir = os.path.split(args.trained_gen)[0]
        trained_gen_args = argparse.Namespace()
        with open(os.path.join(gen_save_dir, 'args.txt'), 'r') as f:
            d = json.load(f)
            for key, value in d.items():
                setattr(trained_gen_args, key, value)
        context_encoder = ConvEncoder(emb_dim=trained_gen_args.emb_dim, hid_dim=trained_gen_args.n_filters, output_dim=trained_gen_args.fea_dim,
                                      kernel_size=trained_gen_args.kernel_size, stride=trained_gen_args.stride, dropout=trained_gen_args.dropout, sent_len3=trained_gen_args.sent_len3)
        model = Generator(vocab_size=len(texts.vocab), emb_dim=trained_gen_args.emb_dim, discriminator=discriminator,
                          context_encoder=context_encoder, agg_output_dim=trained_gen_args.n_filters, lstm_hid_dim=trained_gen_args.hid_dim,
                          lstm_input_dim=trained_gen_args.emb_dim + trained_gen_args.hid_dim, n_layers=n_layers, dropout=trained_gen_args.dropout, K=K, device=device).to(device)
        model.encoder_decoder.load_state_dict(torch.load(args.trained_gen))
        bleu = BLEU(exclude_indices={PAD_IDX, texts.vocab.stoi[texts.eos_token], texts.vocab.stoi[texts.init_token]}) if \
                args.bleu else None
        dist = distinct if args.dist else None
        output_file = open(os.path.join(gen_save_dir, 'inference_outputs.txt'), 'w')

        test_metrics = test_generator(model, dataset['test_iterator'], criterion=nn.CrossEntropyLoss(),
                                      text_field=texts, bleu=bleu, dist=dist, file=output_file)
        output_file.close()
        print_metrics(test_metrics)

if __name__ == '__main__':
    main()
