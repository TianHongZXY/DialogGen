class DataIterator(object):
    def __init__(self, data, batch_size, sampler=None):
        self.data = data
        self.batch_size = batch_size
        self.sampler = sampler

    def batch(self):
        minibatch, size_so_far = [], 0
        for ex in self.data:
            minibatch.append(ex)
            if len(minibatch) == self.batch_size:
                yield minibatch
                minibatch = []
        if minibatch:
            yield minibatch

    def batch_with_sampler(self):
        if self.sampler is None:
            raise ValueError("sampler is not given!")
        for indices in self.sampler:
            yield [self.data[i] for i in indices]
