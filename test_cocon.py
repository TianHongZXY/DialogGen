from seq2seq import Generator, Discriminator, ConvEncoder
from prepare_data import load_cocon_data as load_data
import math
from allennlp.training.metrics import BLEU
from tqdm import tqdm
import time
import torch
import argparse
import torch.nn.functional as F
from train_cocon import disentangling_loss


def write_to_file(file, src_idx, gold_idx, pred_idx, vocab, pad_idx):
    batch_size = src_idx.size(0)
    for i in range(batch_size):
        src_str = ' '.join(list(map(lambda x: vocab.itos[x] if x != pad_idx else '', src_idx[i])))
        gold_str = ' '.join(list(map(lambda x: vocab.itos[x] if x != pad_idx else '', gold_idx[i])))
        pred_str = ' '.join(list(map(lambda x: vocab.itos[x] if x != pad_idx else '', pred_idx[i])))
        file.write('source: ' + src_str + '\n')
        file.write('gold: ' + gold_str + '\n')
        file.write('pred: ' + pred_str + '\n')


def test_generator(model, iterator, text_field, bleu=None):
    model.eval()
    f = open('test_outputs.txt', 'a')
    with torch.no_grad():
        for batch in tqdm(iterator):
            src = batch.src
            # trg = [trg len, batch size]
            trg = batch.trg

            # output = [trg len, batch size, output dim]
            output = model(src, trg, 0)  # turn off teacher forcing
            gold = trg.T
            pred = output.argmax(-1).T[:, 1:]
            write_to_file(f, src_idx=src.T, gold_idx=gold, pred_idx=pred,
                          vocab=text_field.vocab, pad_idx=text_field.vocab.stoi[text_field.pad_token])
            if bleu:
                bleu(predictions=pred, gold_targets=gold)
    if bleu:
        return {'bleu': bleu.get_metric(reset=True)['BLEU']}


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

    return {'epoch_loss_DeCorr': epoch_loss_DeCorr / (len(iterator) - passed_batch),
            'epoch_topic_acc': t_correct / ((num_batches - passed_batch) * batch_size),
            'epoch_person_acc': p_correct / ((num_batches - passed_batch) * batch_size),
            }


def main():
    parser = argparse.ArgumentParser(
        description='CoCon'
    )

    parser.add_argument('--gpu', default=-1, type=int, help='which GPU to use, -1 means using CPU')
    parser.add_argument('--model', default='disc', type=str,
                        help='choose to test which model: {`disc`, `gen`}, (default: `disc`)')

    args = parser.parse_args()
    device = torch.device(args.gpu if (torch.cuda.is_available() and args.gpu >= 0) else 'cpu')
    batch_size = 256
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
    sent_len1 = maxlen = 29  # dailydialog数据集平均每条src+trg有29个单词
    sent_len2 = math.floor((sent_len1 - kernel_size) / stride) + 1
    sent_len3 = math.floor((sent_len2 - kernel_size) / stride) + 1
    dataset = load_data(train_file_path='cocon_test_full.tsv', batch_size=batch_size, device=device,
                        maxlen=maxlen)
    texts = dataset['fields'][0]
    vocab_size = 22079  # len(texts.vocab)
    PAD_IDX = texts.vocab.stoi[texts.pad_token]
    discriminator = Discriminator(vocab_size=vocab_size, emb_dim=embed_dim, padding_idx=PAD_IDX,
                                  feature_dim=feature_dim, n_filters=n_filters, kernel_size=kernel_size,
                                  stride=stride, dropout=dropout, sent_len3=sent_len3).to(device)
    discriminator.load_state_dict(torch.load('models/best-disc-model.pt'))
    if args.model == 'disc':
        test_metrics = test_discriminator(model=discriminator, iterator=dataset['test_iterator'])
        test_loss_decorr = test_metrics['epoch_loss_DeCorr']
        test_topic_acc = test_metrics['epoch_topic_acc']
        test_person_acc = test_metrics['epoch_person_acc']
        print(f'Test Loss DeCorr: {test_loss_decorr:.3f} | '
              f'Test topic acc: {test_topic_acc:.3f} | Test person acc: {test_person_acc:.3f} | ')

    elif args.model == 'gen':
        context_encoder = ConvEncoder(emb_dim=embed_dim, hid_dim=n_filters, output_dim=feature_dim,
                                      kernel_size=kernel_size, stride=stride, dropout=dropout, sent_len3=sent_len3)
        model = Generator(vocab_size=vocab_size, emb_dim=embed_dim, discriminator=discriminator,
                          context_encoder=context_encoder, agg_output_dim=agg_output_dim, lstm_hid_dim=lstm_hid_dim,
                          lstm_input_dim=lstm_input_dim, n_layers=n_layers, dropout=dropout, K=K, device=device).to(device)
        model.encoder_decoder.load_state_dict(torch.load('models/best-enc-dec-model.pt'))
        bleu = BLEU(exclude_indices={PAD_IDX, texts.vocab.stoi[texts.eos_token], texts.vocab.stoi[texts.init_token]})
        test_metrics = test_generator(model, dataset['test_iterator'], text_field=texts, bleu=bleu)
        test_bleu = test_metrics['bleu']
        print(f'Test BLEU: {test_bleu:.5f} |')


if __name__ == '__main__':
    main()
