from my_transformer import make_model
import math
import os
import time
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import numpy as np
from torchtext import data, datasets
from utils import count_parameters, write_metrics, print_metrics, subsequent_mask, write_metrics_to_writer
import json
from datetime import datetime
from dateutil import tz, zoneinfo
import argparse
from tqdm import tqdm
from prepare_data import load_seq2seq_dataset as load_dataset
from prepare_data import load_iwslt
from allennlp.training.metrics import BLEU
from metrics import distinct
from torch.utils.tensorboard import SummaryWriter
from optim.Optim import NoamOptimWrapper as NoamOpt
import logging
import coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


def make_std_mask(tgt, pad):
    "创建一个掩盖pad和future words的mask"
    #  tgt_mask shape = [nbatches, 1, T]
    tgt_mask = (tgt != pad).unsqueeze(-2)
    #  subsequent_mask shape = [1, T, T]
    #  tgt_mask shape = [nbatches, T, T]
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


class Batch(object):
    "存储one batch的数据和mask用于训练"
    def __init__(self, src, tgt, src_padding_idx, tgt_padding_idx):
        #  src shape = [nbatches, T]
        self.src = src
        #  src_mask shape = [nbatches, 1, T]
        self.src_mask = (src != src_padding_idx).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            # self.tgt_y 是 self.tgt往右偏移一位
            self.tgt_y = tgt[:, 1:]
            # tgt_mask shape = [nbatches, T, T]
            self.tgt_mask = make_std_mask(self.tgt, tgt_padding_idx)
            self.ntokens = (self.tgt_y != tgt_padding_idx).data.sum()


def run_epoch(data_iter, model, loss_compute, fields, train=True, writer=None):
    "Training和记录log"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_ce_loss = 0
    tokens = 0
    src_padding_idx = fields['src'].vocab.stoi[fields['src'].pad_token]
    tgt_padding_idx = fields['tgt'].vocab.stoi[fields['tgt'].pad_token]
    tgt_init_token_idx = fields['tgt'].vocab.stoi[fields['tgt'].init_token]
    tgt_eos_token_idx = fields['tgt'].vocab.stoi[fields['tgt'].eos_token]
    hyps = []
    bleu = BLEU(exclude_indices={tgt_padding_idx, tgt_eos_token_idx, tgt_init_token_idx})
    global_step = 0
    for i, batch in enumerate(data_iter):
        setattr(batch, 'tgt_y', batch.tgt[:, 1:])
        batch.tgt = batch.tgt[:, :-1]
        setattr(batch, 'src_mask', (batch.src != src_padding_idx).unsqueeze(-2))
        # tgt_mask shape = [nbatches, T, T]
        setattr(batch, 'tgt_mask', make_std_mask(batch.tgt, tgt_padding_idx))
        setattr(batch, 'ntokens', (batch.tgt_y != tgt_padding_idx).data.sum())
        out = model(batch.src, batch.tgt,
                    batch.src_mask, batch.tgt_mask)
        loss, ce_loss, softmax_logits = loss_compute(out, batch.tgt_y, batch.ntokens)
        if not train:
            pred = softmax_logits.argmax(-1)
            bleu(predictions=pred, gold_targets=batch.tgt_y)
            hyps.extend([h for h in pred.numpy()])
        total_loss += loss
        total_ce_loss += ce_loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        global_step = loss_compute.opt._step if loss_compute.opt else 0
        if i % 100 == 0:
            elapsed = time.time() - start
            if train:
                print("Global Step: %d\tEpoch Step: %d\tLoss: %f\tTokens per Sec: %f\tlr: %f" %
                        (global_step, i + 1, loss / batch.ntokens, tokens / elapsed, loss_compute.opt.rate()))
                if writer:
                    writer.add_scalar('train_kl_div_loss', loss / batch.ntokens, global_step)
                    writer.add_scalar('train_ce_loss', ce_loss / batch.ntokens, global_step)
                    writer.add_scalar('train_ppl', math.exp(ce_loss / batch.ntokens), global_step)
            tokens = 0
            start = time.time()

    metrics = {'kl_div_loss': total_loss / total_tokens, 'ce_loss': total_ce_loss / total_tokens,
               'ppl': math.exp(total_ce_loss / total_tokens)}
    if not train:
        exclude_tokens = {tgt_padding_idx, tgt_eos_token_idx, tgt_init_token_idx}
        inter_dist1, inter_dist2 = distinct(hyps, exclude_tokens=exclude_tokens)
        metrics['bleu'] =  bleu.get_metric(reset=True)['BLEU']
        metrics['dist-1'] = inter_dist1
        metrics['dist-2'] = inter_dist2
    return metrics


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].embedding_size, factor=2, warmup=4000,
            optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), epos=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing using KL div loss"
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.ce_criterion = nn.NLLLoss(reduction='sum', ignore_index=padding_idx)
        # self.ce_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=padding_idx)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        #  先把true_dist的元素全填为smoothing / (size - 2) 的值，-2是因为真实标签位置和padding位置的概率都要另设
        true_dist.fill_(self.smoothing / (self.size - 2))
        #  再把true_dist在列上以target为索引的地方的值变为confidence
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        #  把padding的位置概率都变为0
        true_dist[:, self.padding_idx] = 0
        #  把target就预测padding token的整个概率分布都设为0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        #  用true_dist这个经过平滑的概率分布去替代target的one-hot概率分布是为了避免模型的预测分布也向one-hot靠近
        #  避免模型变得太过confident，模型学着预测时变得更不确定，这对ppl有伤害，但是能够提升预测的准确性和BLEU分数
        #  当x的概率分布很尖锐时，loss将变大
        return self.criterion(x, Variable(true_dist, requires_grad=False)), self.ce_criterion(x, target.long())


class SimpleLossCompute:
    "A simple loss compute and train function"
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss, ce_loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss /= norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm, ce_loss.item(), x.detach().cpu()


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    #  ys shape [nbatches, T_q] = [1, 1]
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        #  out shape = [nbatches, T_q, d_model]
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        #  prob shape = [nbatches, vocab]
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        #  ys shape = [nbatches, T_q + 1]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch_size, 10)))
        data[:, 0] = 1
        src = Variable(data[:, 1:], requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                #  生成一个100倍batch_size的batch，实际上是预加载100个batch出来以加快下面的load速度
                for p in data.batch(d, self.batch_size * 100):
                    #  对预先加载出来的100个batch按照sort_key排序，然后按正常batch_size一个一个生成minibatch
                    p_batch = data.batch(p, self.batch_size)
                    #  对样本长度都相近的minibatch做一次shuffle
                    # for b in p_batch:
                    #     yield b
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(padding_idx, batch):
    "Fix order in torchtext to match ours"
    #  torchtext生成的batch中seq shape = [T, nbatches]，转置一下获得我们想要的形状[nbatches, T]
    src, tgt = batch.src.transpose(0, 1), batch.tgt.transpose(0, 1)
    return Batch(src, tgt, padding_idx)


def main():
    parser = argparse.ArgumentParser(
        description='transformer'
    )

    parser.add_argument('--gpu', default=-1, type=int, help='which GPU to use, -1 means using CPU')
    parser.add_argument('--save', default=0, type=int, help='whether to save model or not')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--d_model', default=512, type=int, help='hidden dim of lstm')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout ratio')
    parser.add_argument('--n_epochs', default=30, type=int, help='num of train epoch')
    parser.add_argument('--clip', default=None, type=float, help='grad clip')
    parser.add_argument('--teach', default=1, type=float, help='propability of using teacher')
    parser.add_argument('--maxlen', default=29, type=int, help='fixed length of text')
    parser.add_argument('--train_file', default=None, type=str, help='train file path')
    parser.add_argument('--valid_file', default=None, type=str, help='valid file path')
    parser.add_argument('--test_file', default=None, type=str, help='test file path')
    parser.add_argument('--save_dir', default='models', type=str, help='save dir')
    parser.add_argument('--num_workers', default=0, type=int, help='how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('--vocab_file', default=None, type=str, help='vocab file path')
    parser.add_argument('--lr', default=0, type=float, help='learning rate')
    parser.add_argument('--l2', default=0, type=float, help='l2 regularization')
    parser.add_argument('--warmup', default=4000, type=int, help='warmup step for learning rate')
    parser.add_argument('--min_freq', default=2, type=int, help='min freq for word not to be converted into <unk>')
    parser.add_argument('--n_layers', default=6, type=int, help='layers of transformer')
    parser.add_argument('--head', default=8, type=int, help='num of heads of multi-head attention')
    parser.add_argument('--d_ff', default=2048, type=int, help='dim of FFN')
    parser.add_argument('--smoothing', default=0.1, type=float, help='smoothing rate of computing kl div loss')
    parser.add_argument('--factor', default=1, type=float, help='factor of learning rate')
    parser.add_argument('--model_path', default=None, type=str, help='restore model to continue training')
    parser.add_argument('--global_step', default=None, type=int, help='global step for continuing training')
    parser.add_argument('--share_decoder_embeddings', default=0, type=int, help="whether share decoder's embedding with generator's pre-softmax matrix")
    args, unparsed = parser.parse_known_args()
    device = torch.device(args.gpu if (torch.cuda.is_available() and args.gpu >= 0) else 'cpu')
    writer = None
    if args.save and args.model_path is None:
        tz_sh = tz.gettz('Asia/Shanghai')
        save_dir = os.path.join(args.save_dir, 'run' + str(datetime.now(tz=tz_sh)).replace(":", "-").split(".")[0].replace(" ", '.'))
        args.save_dir = save_dir
        logger.info(f"Saving log files and model to {save_dir}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    elif args.save and args.model_path:
        save_dir = os.path.split(args.model_path)[0]

    if args.save:
        writer = SummaryWriter(os.path.join(save_dir, 'summary'))
    ########################################################
    # train the model
    logger.info("Loading train dataset...")
    train_dict = load_dataset(args, file_path='.', train_file=args.train_file, valid_file=args.valid_file, test_file=args.test_file, train=True)
    # logger.info("Loading valid dataset...")
    # valid_dict = load_dataset(args, file_path='.', train_file=args.train_file, valid_file=args.valid_file, test_file=args.test_file, train=False)
    # test_dict = load_dataset(args, args.test_file, train=False)
    train_iter = MyIterator(train_dict['dataset'][0], batch_size=args.bs, train=True,
            sort_key=lambda x: (len(x.src), len(x.tgt)), device=device, repeat=False, sort=True)
    # train_iter = data.BucketIterator(train_dict['dataset'], batch_size=args.bs, train=True,
    #         sort_key=lambda x: (len(x.src), len(x.tgt)), device=device, repeat=False, sort=True)
    valid_iter = MyIterator(train_dict['dataset'][1], batch_size=args.bs, train=False,
            sort_key=lambda x: (len(x.src), len(x.tgt)), device=device, repeat=False, sort=True)
    # valid_iter = data.BucketIterator(valid_dict['dataset'], batch_size=args.bs, train=False,
    #         sort_key=lambda x: (len(x.src), len(x.tgt)), device=device, repeat=False, sort=True)

    test_iter = MyIterator(train_dict['dataset'][2], batch_size=args.bs, train=False,
            sort_key=lambda x: (len(x.src), len(x.tgt)), device=device, repeat=False, sort=True)
    SRC = train_dict['src']
    TGT = train_dict['tgt']
    """
    ########################################################
    # test model on iwslt
    dataset_dict = load_iwslt()
    train_iter = data.BucketIterator(dataset_dict['train'], batch_size=args.bs, train=True,
            sort_key=lambda x: (len(x.src), len(x.tgt)), device=device, repeat=False, sort=True)
    valid_iter = data.BucketIterator(dataset_dict['val'], batch_size=args.bs, train=False,
            sort_key=lambda x: (len(x.src), len(x.tgt)), device=device, repeat=False, sort=True)
    test_iter = data.BucketIterator(dataset_dict['test'], batch_size=args.bs, train=False,
            sort_key=lambda x: (len(x.src), len(x.tgt)), device=device, repeat=False, sort=True)
    SRC = dataset_dict['src']
    TGT = dataset_dict['tgt']
    """
    padding_idx = TGT.vocab.stoi[TGT.pad_token]

    model = make_model(len(SRC.vocab), len(TGT.vocab), N=args.n_layers,
            d_model=args.d_model, d_ff=args.d_ff, h=args.head, dropout=args.dropout)
    if args.model_path is not None:
        logger.info(f"Restore model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location={'cuda:0': 'cuda:' + str(args.gpu)}))
    if args.share_decoder_embeddings:
        logger.info("The model shares tgt embedding with generator proj.")
        model.generator.proj.weight = model.tgt_embed[0].lut.weight

    print(f'The model has {count_parameters(model)} trainable parameters')
    model.to(device)
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=padding_idx, smoothing=args.smoothing)
    criterion.to(device)
    model_opt = NoamOpt(args.d_model, args.factor, args.warmup,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    if args.global_step is not None:
        logger.info(f'Global step start from {args.global_step}')
        model_opt._step = args.global_step

    best_step = -1
    best_valid_loss = float('inf')
    best_ppl = float('inf')
    for epoch in range(args.n_epochs):
        model.train()
        train_metrics = run_epoch(train_iter, model, SimpleLossCompute(model.generator, criterion, opt=model_opt),
                                  fields={'src': SRC, 'tgt': TGT}, writer=writer)
        print_metrics(train_metrics, mode='Train')
        global_step = model_opt._step
        model.eval()
        valid_metrics = run_epoch(valid_iter, model, SimpleLossCompute(model.generator, criterion, opt=None),
                                  fields={'src': SRC, 'tgt': TGT}, train=False)
        test_metrics = run_epoch(test_iter, model, SimpleLossCompute(model.generator, criterion, opt=None),
                                  fields={'src': SRC, 'tgt': TGT}, train=False)

        if valid_metrics['ce_loss'] < best_valid_loss:
            best_valid_loss = valid_metrics['ce_loss']
            best_ppl = valid_metrics['ppl']
            best_step = global_step
            valid_metrics['best_step'] = best_step
            valid_metrics['best_valid_loss'] = best_valid_loss
            valid_metrics['best_ppl'] = best_ppl

        print_metrics(valid_metrics, mode='Valid')
        print_metrics(test_metrics, mode='Test')
        if args.save:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_global_step-{global_step}.pt'))
            write_metrics_to_writer(valid_metrics, writer, global_step, mode='Valid')
            write_metrics_to_writer(test_metrics, writer, global_step, mode='Test')
            with open(os.path.join(save_dir, f'log_global_step-{global_step}.txt'), 'w') as log_file:
                log_file.write(f'Global Step: {global_step}\n')
                write_metrics(train_metrics, log_file, mode='Train')
                write_metrics(valid_metrics, log_file, mode='Valid')
                write_metrics(test_metrics, log_file, mode='Test')

if __name__ == '__main__':
    main()

#  if __name__ == '__main__':
    #  =========================================================================
    # Test load dataset
    #  resdict = load_dataset('/data/zxy/DialogGen/metalwoz-v1/allennlp_test.tsv')
    #  src = resdict['src']
    #  print(src.vocab.stoi)
    #  tgt = resdict['tgt']
    #  print(tgt.vocab.stoi)
    #  train_iter = MyIterator(resdict['dataset'], batch_size=10, train=True,
    #                          repeat=False, sort=True, sort_key=lambda x: (len(x.src), len(x.tgt)))
    #  for i, batch in enumerate(train_iter):
    #      setattr(batch, 'tgt_y', batch.tgt[:, 1:])
    #      batch.tgt = batch.tgt[:, :-1]
    #      setattr(batch, 'src_mask', (batch.src != src.vocab.stoi[src.pad_token]).unsqueeze(-2))
    #      # tgt_mask shape = [nbatches, T, T]
    #      setattr(batch, 'tgt_mask', make_std_mask(batch.tgt, tgt.vocab.stoi[tgt.pad_token]))
    #      setattr(batch, 'ntokens', (batch.tgt_y != tgt.vocab.stoi[tgt.pad_token]).data.sum())
    #      print('batch', batch)
    #      print('src', batch.src.size(), batch.src)
    #      print('tgt', batch.tgt.size(), batch.tgt)
    #      print('tgt_y', batch.tgt_y.size(), batch.tgt_y)
    #      print('src_mask', batch.src_mask.size(), batch.src_mask)
    #      print('tgt_mask', batch.tgt_mask.size(), batch.tgt_mask)
    #      break
        # if i == 3:
        #     break
    #  =========================================================================
    # Train the simple copy task.
    #  V = 11
    #  criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    #  model = make_model(V, V, N=1, d_model=10, d_ff=20, h=2)
    #  model_opt = NoamOpt(model.src_embed[0].d_model, 1, 10,
    #          torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9))
    #  for epoch in range(1):
    #      model.train()
    #      run_epoch(data_gen(V, 2, 2), model, SimpleLossCompute(model.generator, criterion, model_opt))
    #      model.eval()
    #      print(run_epoch(data_gen(V, 2, 2), model,
    #                      SimpleLossCompute(model.generator, criterion, None)))
    #  =========================================================================
    #  测试greedy_decode
    
    #  model.eval()
    #  src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]))
    #  src_mask = Variable(torch.ones(1, 1, 10))
    #  print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
    #  for batch in data_gen(V, batch_size=1, nbatches=1):
    #      greedy_decode(model, batch.src, batch.src_mask, 3, 1)
    #  raise ValueError()
