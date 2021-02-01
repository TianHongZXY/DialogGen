import torch.utils.data as data
import torch
import random
import numpy as np
import pandas as pd
from collections import namedtuple, defaultdict


class DatasetFromTSV(data.Dataset):
    def __init__(self, tsv_path):
        self.data = pd.read_csv(tsv_path, delimiter='\t', header=None)
        self.src = np.asarray(self.data.iloc[:, 0])
        self.tgt = np.asarray(self.data.iloc[:, 1])
        assert self.src.shape[0] == self.tgt.shape[0]
        # self.res = namedtuple('res', ['src', 'tgt'])

    def __getitem__(self, index):
        # return self.res(src=self.src[index], tgt=self.tgt[index])
        return self.src[index], self.trg[index]

    def __len__(self):
        return self.src.shape[0]

    def shuffle(self):
        random.seed(20020206)
        random.shuffle(self.src)
        random.seed(20020206)
        random.shuffle(self.tgt)


class Dataset(data.Dataset):
    def __init__(self, data_source):
        self.data_source = data_source

    def __getitem__(self, index):
        return self.data_source[index]

    def __len__(self):
        return len(self.data_source)

    def shuffle(self):
        random.shuffle(self.data_source)


def compute_len(x, padding_idx=0):
    return np.sum(x != padding_idx)


class BucketSamplerv1(data.Sampler):
    def __init__(self, lengths, shuffle=True, batch_size=32, drop_last=False):
        super().__init__(lengths)
        self.lengths = lengths
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.index = np.argsort(lengths)

    def __iter__(self):
        bucket = list(torch.split(torch.tensor(self.index, dtype=torch.int, device='cpu'), self.batch_size))
        if len(bucket) > 1 and self.drop_last:
            if len(bucket[-1]) < len(bucket[-2]):
                bucket = bucket[:-1]
        if self.shuffle:
            random.shuffle(bucket)
        for batch in bucket:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore


class BucketSamplerv2(data.Sampler):
    def __init__(self, lengths, buckets=(1, 500, 1), shuffle=True, batch_size=32, drop_last=False):
        super().__init__(lengths)

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last

        assert isinstance(buckets, tuple)
        bmin, bmax, bstep = buckets
        assert (bmax - bmin) % bstep == 0

        buckets = defaultdict(list)
        for i, length in enumerate(lengths):
            if length > bmin:
                bucket_size = min((length // bstep) * bstep, bmax)
                buckets[bucket_size].append(i)

        self.buckets = dict()
        for bucket_size, bucket in buckets.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket, dtype=torch.int, device='cpu')

        # call __iter__() to store self.length
        self.__iter__()

    def __iter__(self):
        if self.shuffle == True:
            for bucket_size in self.buckets.keys():
                self.buckets[bucket_size] = self.buckets[bucket_size][
                    torch.randperm(self.buckets[bucket_size].nelement())]

        batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket, self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last == True:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            batches += curr_bucket

        self.length = len(batches)

        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    import time
    src = np.array([
                *[[1,2,3,4,0,0,0] for _ in range(1000)],
                *[[1,2,3,0,0,0,0] for _ in range(1000)],
                *[[1,2,3,4,5,0,0] for _ in range(1000)],
                *[[1,2,3,4,5,6,0] for _ in range(150)],
                *[[1,2,3,4,5,6,7] for _ in range(1000)],
                *[[1,2,0,0,0,0,0] for _ in range(500)],
                *[[1,0,0,0,0,0,0] for _ in range(700)],
                ])
    np.random.shuffle(src)
    dataset = Dataset(src)
    # dataset = DatasetFromTSV('/data/zxy/DialogGen/transformer_models/tiny_train.tsv')
    sampler = data.SequentialSampler(data_source=dataset)
    bucket_samplerv1 = BucketSamplerv1([np.sum(x != 0) for x in src], batch_size=32)
    bucket_samplerv2 = BucketSamplerv2([np.sum(x != 0) for x in src], batch_size=32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=bucket_samplerv2, num_workers=0)
    start = time.time()
    for i, b in enumerate(dataloader):
        print(b)
    elapsed = time.time() - start
    print(f'time {elapsed}')
