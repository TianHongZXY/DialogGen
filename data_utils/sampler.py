import torch.utils.data as data
import numpy as np
import torch
from collections import defaultdict
import random


def compute_len(x, padding_idx=0):
    return np.sum(x != padding_idx)


class BucketSamplerv1(data.Sampler):
    def __init__(self, lengths, sort_key, shuffle=True, batch_size=32, drop_last=False):
        super().__init__(lengths)
        self.data_source = self.lengths = lengths
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.index = [i[0] for i in sorted(enumerate(self.lengths), key=sort_key)]

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
            return len(self.lengths) // self.batch_size  # type: ignore
        else:
            return (len(self.lengths) + self.batch_size - 1) // self.batch_size  # type: ignore


class BucketSamplerv2(data.Sampler):
    def __init__(self, lengths, buckets=(1, 500, 1), shuffle=True, batch_size=32, drop_last=False):
        super().__init__(lengths)
        self.data_source = lengths
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last

        assert isinstance(buckets, tuple)
        bmin, bmax, bstep = buckets
        assert (bmax - bmin) % bstep == 0

        buckets = defaultdict(list)
        for i, length in enumerate(lengths):
            if length >= bmin:
                bucket_size = min((length // bstep) * bstep, bmax)
                buckets[bucket_size].append(i)

        self.buckets = dict()
        for bucket_size, bucket in buckets.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket, dtype=torch.int, device='cpu')

        # call __iter__() to store self.length
        self.__iter__()

    def __iter__(self):
        if self.shuffle:
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
