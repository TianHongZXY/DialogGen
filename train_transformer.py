from my_transformer import subsequent_mask, make_model
import math
import os
import time
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import numpy as np
from torchtext import data, datasets
from utils import count_parameters, write_metrics, print_metrics
import json
from datetime import datetime
from dateutil import tz, zoneinfo
import argparse
from tqdm import tqdm
from memory_profiler import profile


def make_std_mask(tgt, pad):
    "创建一个掩盖pad和future words的mask"
    #  tgt_mask shape = [nbatches, 1, T]
    tgt_mask = (tgt != pad).unsqueeze(-2)
    #  subsequent_mask shape = [1, T, T]
    #  tgt_mask shape = [nbatches, T, T]
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask


class Batch(object):
    "存储one batch的数据和mask用于训练"
    def __init__(self, src, tgt, src_padding_idx, tgt_padding_idx):
        #  src shape = [nbatches, T]
        self.src = src
        #  src_mask shape = [nbatches, 1, T]
        self.src_mask = (src != src_padding_idx).unsqueeze(-2)
        if tgt is not None:
            #  丢掉句子末尾标记<eos>，用于作训练的输入
            self.tgt = tgt[:, :-1]
            #  丢掉句子开头标记<sos>，用于作训练的标签，即输入往右偏移一位
            self.tgt_y = tgt[:, 1:]
            # tgt_mask shape = [nbatches, T, T]
            self.tgt_mask = make_std_mask(self.tgt, tgt_padding_idx)
            self.ntokens = (self.tgt_y != tgt_padding_idx).data.sum()


def run_epoch(data_iter, model, loss_compute, fields, train=True):
    "Training和记录log"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_ce_loss = 0
    tokens = 0
    src_padding_idx = fields['src'].vocab.stoi[fields['src'].pad_token]
    tgt_padding_idx = fields['tgt'].vocab.stoi[fields['tgt'].pad_token]
    for i, batch in enumerate(data_iter):
        setattr(batch, 'tgt_y', batch.tgt[:, 1:])
        batch.tgt = batch.tgt[:, :-1]
        setattr(batch, 'src_mask', (batch.src != src_padding_idx).unsqueeze(-2))
        # tgt_mask shape = [nbatches, T, T]
        setattr(batch, 'tgt_mask', make_std_mask(batch.tgt, tgt_padding_idx))
        setattr(batch, 'ntokens', (batch.tgt_y != tgt_padding_idx).data.sum())
        out = model(batch.src, batch.tgt,
                    batch.src_mask, batch.tgt_mask)
        loss, ce_loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_ce_loss += ce_loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1 and train:
            elapsed = time.time() - start
            print("Epoch Step: %d\tLoss: %f\tTokens per Sec: %f\tlr: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed, loss_compute.opt.rate()))
            start = time.time()
            tokens = 0
    return {'kl_div_loss': total_loss / total_tokens, 'ce_loss': total_ce_loss / total_tokens,
            'ppl': math.exp(total_ce_loss / total_tokens)}


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Implement `lrate` = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
        lr 最大是d_model^(-0.5) * warmup_steps^(-0.5) = (d_model * warmup_steps)^(-0.5)
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                min(step **(-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, factor=2, warmup=4000,
            optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), epos=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing using KL div loss"
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.ce_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=padding_idx)
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
        return loss.item() * norm, ce_loss.item()

#  from pysnooper import snoop
#  from torchsnooper import snoop
#  @snoop()
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
                    for b in p_batch:
                        yield b
                    # for b in random_shuffler(list(p_batch)):
                    #     yield b
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


def load_dataset(args, path):
    import spacy
    spacy_en = spacy.load('en')
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    SRC = data.Field(tokenize=tokenize_en, pad_token=pad_token, batch_first=True)
    TGT = data.Field(tokenize=tokenize_en, init_token=sos_token,
            eos_token=eos_token, pad_token=pad_token, batch_first=True)
    MAX_LEN = args.maxlen
    fields = [('src', SRC), ('tgt', TGT)]
    dataset = data.TabularDataset(path, 'TSV', fields=fields,
            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['tgt']) <= MAX_LEN)
    MIN_FREQ = args.min_freq
    SRC.build_vocab(dataset.src, min_freq=MIN_FREQ)
    TGT.build_vocab(dataset.tgt, min_freq=MIN_FREQ)
    return {'dataset': dataset, 'src': SRC, 'tgt': TGT}


def main():
    parser = argparse.ArgumentParser(
        description='transformer'
    )

    parser.add_argument('--gpu', default=-1, type=int, help='which GPU to use, -1 means using CPU')
    parser.add_argument('--save', default=False, type=bool, help='whether to save model or not')
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
    parser.add_argument('--l2', default=0, type=float, help='l2 regularization')
    parser.add_argument('--warmup', default=4000, type=int, help='warmup step for learning rate')
    parser.add_argument('--min_freq', default=2, type=int, help='min freq for word not to be converted into <unk>')
    parser.add_argument('--n_layers', default=6, type=int, help='layers of transformer')
    parser.add_argument('--head', default=8, type=int, help='num of heads of multi-head attention')
    parser.add_argument('--d_ff', default=2048, type=int, help='dim of FFN')
    args, unparsed = parser.parse_known_args()
    device = torch.device(args.gpu if (torch.cuda.is_available() and args.gpu >= 0) else 'cpu')
    if args.save:
        tz_sh = tz.gettz('Asia/Shanghai')
        save_dir = os.path.join(args.save_dir, 'run' + str(datetime.now(tz=tz_sh)).replace(":", "-").split(".")[0].replace(" ", '.'))
        args.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    train_dict = load_dataset(args, args.train_file)
    valid_dict = load_dataset(args, args.valid_file)
    # test_dict = load_dataset(args, args.test_file)
    # train_iter = MyIterator(train_dict['dataset'], batch_size=args.bs, train=True,
    #         sort_key=lambda x: (len(x.src), len(x.tgt)), device=device, repeat=False, sort=True)
    train_iter = data.BucketIterator(train_dict['dataset'], batch_size=args.bs, train=True,
            sort_key=lambda x: (len(x.src), len(x.tgt)), device=device, repeat=False, sort=True)
    valid_iter = data.BucketIterator(valid_dict['dataset'], batch_size=args.bs, train=False,
            sort_key=lambda x: (len(x.src), len(x.tgt)), device=device, repeat=False, sort=True)
    # test_iter = MyIterator(test_dict['dataset'], batch_size=args.bs, train=False,
    #         sort_key=lambda x: (len(x.src), len(x.tgt)), device=device, repeat=False, sort=True)
    SRC = train_dict['src']
    TGT = train_dict['tgt']
    padding_idx = TGT.vocab.stoi[TGT.pad_token]

    model = make_model(len(SRC.vocab), len(TGT.vocab), N=args.n_layers,
            d_model=args.d_model, d_ff=args.d_ff, h=args.head, dropout=args.dropout)
    print(f'The model has {count_parameters(model)} trainable parameters')
    model.to(device)
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=padding_idx, smoothing=0.1)
    criterion.to(device)
    model_opt = NoamOpt(args.d_model, 1, args.warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e9))
    for epoch in range(args.n_epochs):
        model.train()
        train_metrics = run_epoch(train_iter, model, SimpleLossCompute(model.generator, criterion, opt=model_opt),
                                  fields={'src': SRC, 'tgt': TGT})
        print_metrics(train_metrics, mode='Train')
        #  print(f"train loss = {result_dict['kl_div_loss']}\t train ppl = {math.exp(result_dict['ce_loss'])}")
        model.eval()
        valid_metrics = run_epoch(valid_iter, model, SimpleLossCompute(model.generator, criterion, opt=None),
                                  fields={'src': SRC, 'tgt': TGT})
        print_metrics(valid_metrics, mode='Valid')
        #  print(f"valid loss = {result_dict['kl_div_loss']}\t valid ppl = {math.exp(result_dict['ce_loss'])}")
        # test_metrics = run_epoch(test_iter, model, SimpleLossCompute(model.generator, criterion, opt=None),
        #                          fields={'src': SRC, 'tgt': TGT})
        # print_metrics(test_metrics, mode='Test')
        #  print(f"test loss = {result_dict['kl_div_loss']}\t test ppl = {math.exp(result_dict['ce_loss'])}")
        if args.save:
            torch.save(model.state_dict(), os.path.join(save_dir, f'transformer-{epoch}.pt'))
            with open(os.path.join(save_dir, f'log_epoch{epoch}.txt'), 'w') as log_file:
                log_file.write(f'Epoch: {epoch:02}\n')
                write_metrics(train_metrics, log_file, mode='Train')
                write_metrics(valid_metrics, log_file, mode='Valid')
                # write_metrics(test_metrics, log_file, mode='Test')

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
