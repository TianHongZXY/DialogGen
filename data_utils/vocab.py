from collections import defaultdict


class Vocab(object):
    UNK = '<unk>'

    def __init__(self,
                 counter,
                 specials=('<pad>', '<unk>'),
                 min_freq=1,
                 max_size=None):
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        self.itos = list()
        if specials:
            self.itos = list(specials)
            max_size = None if max_size is None else max_size + len(specials)

        for tok in specials:
            del counter[tok]

        # 排序方式先由出现次数决定且降序，然后再按照字典序升序
        words_and_freqs = sorted(counter.items(), key=lambda w_f: w_f[0])
        words_and_freqs.sort(key=lambda w_f: w_f[1], reverse=True)

        # 丢掉freq小于min_freq的单词
        self.itos += [word for word, freq in words_and_freqs if freq >= min_freq]
        if max_size and max_size > 0:
            self.itos = self.itos[:max_size]

        if Vocab.UNK in specials:
            unk_index = specials.index(Vocab.UNK)
            self.unk_index = unk_index
            self.stoi = defaultdict(self._default_unk_index)
        else:
            self.stoi = defaultdict()

        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

    def _default_unk_index(self):
        return self.unk_index

    def lookup_indices(self, tokens):
        indices = [self.__getitem__(token) for token in tokens]
        return indices

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)
