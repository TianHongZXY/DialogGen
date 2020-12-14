from torchtext.data import Field, BucketIterator, TabularDataset, Dataset
import torch
import logging

logger = logging.getLogger(__name__)


def tokenizer(text):
    return text.split()


def load_cocon_data(train_file_path, val_file_path=None, test_file_path=None,
                    batch_size=32, device=torch.device('cpu'), maxlen=29):
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
    texts.build_vocab(train)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
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
                                               train='dailydialog_src_tgt_train.tsv',
                                               validation='dailydialog_src_tgt_val.tsv',
                                               test='dailydialog_src_tgt_test.tsv',
                                               format='tsv',
                                               fields=fields
                                               )
    BATCH_SIZE = batch_size
    SRC.build_vocab(train)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.src),
        sort_within_batch=True,
        device=device)
    return {"fields": (SRC, ), "vocab": SRC.vocab, "train_data": train, "val_data": valid,
            "test_data": test, "train_iterator": train_iterator,
            "valid_iterator": valid_iterator, "test_iterator": test_iterator}
