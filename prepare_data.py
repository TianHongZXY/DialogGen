from torchtext.data import Field, BucketIterator, TabularDataset, Dataset
import torch


def tokenizer(text):
    return text.split()


def load_data(train_file_path='dailydialog_src_tgt_train.tsv', val_file_path=None, test_file_path=None, batch_size=32):
    SRC = Field(tokenize=tokenizer,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
    fields = [("src", SRC), ("trg", SRC)]
    train, valid, test = TabularDataset.splits(path='.',
                                               train='dailydialog_src_tgt_train.tsv',
                                               validation='dailydialog_src_tgt_val.tsv',
                                               test='dailydialog_src_tgt_test.tsv',
                                               format='tsv',
                                               fields=fields
                                               )
    BATCH_SIZE = batch_size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
