from torchtext.data import Field, BucketIterator, TabularDataset, Dataset
from torchtext.vocab import Vectors, GloVe, Vocab
import torch
import logging
import os
import re
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
import logging
import coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


class DatasetFromTSV(Dataset):
    def __init__(self, tsv_path):
        self.data = pd.read_csv(tsv_path, delimiter='\t', header=None)
        self.src = np.asarray(self.data.iloc[:, 0])
        self.trg = np.asarray(self.data.iloc[:, 1])
        self.same_topic = np.asarray(self.data.iloc[:, 2])
        self.same_person = np.asarray(self.data.iloc[:, 3])

    def __getitem__(self, index):
        return self.src[index], self.trg[index], self.same_topic[index], self.same_person[index]

    def __len__(self):
        return self.src.shape[0]


def tokenizer(text):
    text = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', " ", text)
    return text.split()


def load_metalwoz(train_file_path, val_file_path=None, test_file_path=None,
                    batch_size=32, device=torch.device('cpu'), maxlen=29):
    pass


def build_field_vocab(vocab_file, field, vectors=None):
    d = dict()
    with open(vocab_file, 'r') as f:
        for line in f.readlines():
            line = line.split(':')
            word, freq = line[0], line[1].strip('\n')
            d[word] = int(freq)
    freqs = Counter(d)
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]))
    field.vocab = field.vocab_cls(counter=freqs,
                                  max_size=35000,
                                  min_freq=2,
                                  specials=specials,
                                  vectors=vectors)
    return field


def load_cocon_data(args, train_file_path, val_file_path=None, test_file_path=None,
                    batch_size=32, device=torch.device('cpu'), maxlen=29, pretrained_embed_file=None):
    texts = Field(tokenize=tokenizer,
                  init_token='<sos>',
                  eos_token='<eos>',
                  lower=True,
                  fix_length=maxlen)
    topic_label = Field(sequential=False,
                        use_vocab=False,
                        )
    persona_label = Field(sequential=False,
                          use_vocab=False
                          )
    fields = [("src", texts), ("trg", texts), ("same_topic", topic_label), ("same_person", persona_label)]
    dataset = TabularDataset.splits(path='.',
                                   train=train_file_path,
                                   validation=val_file_path,
                                   test=test_file_path,
                                   format='tsv',
                                   fields=fields
                                   )
    if val_file_path is None or test_file_path is None:
        logger.warning("val_file or test_file is None, it will use the train data")
    train = dataset[0]
    valid = None if not val_file_path else dataset[1]
    test = None if not test_file_path else dataset[2]
    if valid is None:
        valid = train
    if test is None:
        test = train
    BATCH_SIZE = batch_size
    # glove.6B.300d.txt为预先下载好的预训练词向量文件
    vectors = None
    if pretrained_embed_file:
        if not os.path.exists('.vector_cache'):
            os.mkdir('.vector_cache')
            vectors = Vectors(name=pretrained_embed_file)

    if args.vocab_file is None and args.trained_disc is None:
        logger.info("No pre-defined vocab given, building new vocab now...")
        # train disc的时候保存vocab文件，到了train gen时加载，保证两者的embedding共用一个vocab
        texts.build_vocab(train,
                          max_size=35000,
                          min_freq=2,
                          vectors=vectors
                          )
        with open(os.path.join(args.save_dir, 'vocab.txt'), 'w') as f:
            for k, v in  texts.vocab.freqs.most_common():
                f.write( "{}:{}\n".format(k, v))
    elif args.vocab_file is not None:
        #  优先使用给定的vocab file
        logger.info(f"Using vocab file from {args.vocab_file}")
        texts = build_field_vocab(vocab_file=args.vocab_file, field=texts, vectors=vectors)
    elif args.trained_disc is not None:
        # 当没有给定vocab file时，选用disc的vocab
        disc_save_dir = os.path.split(args.trained_disc)[0]
        vocab_file = os.path.join(disc_save_dir, 'vocab.txt')
        logger.info(f"Using vocab file from {vocab_file}")
        texts = build_field_vocab(vocab_file=vocab_file, field=texts, vectors=vectors)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
        shuffle=True,
        sort_key=lambda x: len(x.src) + len(x.trg),
        sort_within_batch=True,
        device=device)
    return {"fields": (texts, topic_label, persona_label), "vocab": texts.vocab, "train_data": train, "val_data": valid,
            "test_data": test, "train_iterator": train_iterator,
            "valid_iterator": valid_iterator, "test_iterator": test_iterator}


def load_data(train_file_path='dailydialog_src_tgt_train.tsv', val_file_path=None, test_file_path=None,
              batch_size=32, device=torch.device('cpu')):
    SRC = Field(tokenize=tokenizer,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                include_lengths=True)
    fields = [("src", SRC), ("trg", SRC)]
    train, valid, test = TabularDataset.splits(path='.',
                                               train='src_tgt_train.tsv',
                                               validation='src_tgt_val.tsv',
                                               test='src_tgt_test.tsv',
                                               format='tsv',
                                               fields=fields
                                               )
    BATCH_SIZE = batch_size
    SRC.build_vocab(train)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.src),
        shuffle=True,
        sort_within_batch=True)

    train_iterator = DataLoader(list(train_iterator), batch_size=args.batch_size, num_workers=args.num_workers)
    valid_iterator = DataLoader(list(valid_iterator), batch_size=args.batch_size, num_workers=args.num_workers)
    test_iterator = DataLoader(list(test_iterator), batch_size=args.batch_size, num_workers=args.num_workers)

    return {"fields": (SRC, ), "vocab": SRC.vocab, "train_data": train, "val_data": valid,
            "test_data": test, "train_iterator": train_iterator,
            "valid_iterator": valid_iterator, "test_iterator": test_iterator}


# if __name__ == '__main__':
    # d = dict()
    # with open('models/run2021-01-06.20-19-55/vocab.txt', 'r') as f:
    #     for line in f.readlines():
    #         line = line.split(':')
    #         print(line)
    #         word, freq = line[0], line[1].strip('\n')
    #         d[word] = int(freq)
    #         print(word, freq)
    # train = DatasetFromTSV('metalwoz_v1/test.tsv')
    # train_iterator = DataLoader(train, batch_size=32, shuffle=True, num_workers=6)
    # for i, batch in enumerate(train_iterator):
    #     print(batch[0])
    #     print(batch[1])
    #     print(batch[2])
    #     print(batch[3])
    #     if i == 10:
    #         break
