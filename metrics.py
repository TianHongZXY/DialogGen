#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics class.
"""

from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import json
import argparse


def distinct(seqs):
    """
    Calculate intra/inter distinct 1/2.
    看论文的意思应该是用inter而不是intra
    """
    # batch_size = len(seqs)
    # intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        # intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        # intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))
        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    # intra_dist1 = np.average(intra_dist1)
    # intra_dist2 = np.average(intra_dist2)
    return inter_dist1, inter_dist2  #, intra_dist1, intra_dist2,


def bleu(hyps, refs):
    """ Calculate bleu 1/2/3/4. """
    bleu_1 = []
    bleu_2 = []
    bleu_3 = []
    bleu_4 = []
    for hyp, ref in zip(hyps, refs):
        # bleu1
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        # bleu2
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
        # bleu_3
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1.0/3, 1.0/3, 1.0/3, 0])
        except:
            score = 0
        bleu_3.append(score)
        # bleu_4
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.25, 0.25, 0.25, 0.25])
        except:
            score = 0
        bleu_4.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    bleu_3 = np.average(bleu_3)
    bleu_4 = np.average(bleu_4)
    return bleu_1, bleu_2, bleu_3, bleu_4


def test_cvae(file):
    seqs = []
    refs = []
    with open(file=file, mode='r', encoding='utf-8') as fr:
        for line in fr:
            l = json.loads(line)
            seqs.append(l['result'])
            refs.append(l['response'])
    inter_dist1, inter_dist2 = distinct(seqs)
    bleu_1, bleu_2, bleu_3, bleu_4 = bleu(seqs, refs)
    print(f'inter_dist1: {inter_dist1:.7f} | inter_dist2: {inter_dist2:.7f}')
    print(f'bleu_1: {bleu_1:.7f} | bleu_2: {bleu_2:.7f} | bleu_3: {bleu_3:.7f} | bleu_4: {bleu_4:.7f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='metrics'
    )
    parser.add_argument('--model', default='cvae', type=str, help="which model's results to test")
    parser.add_argument('--file', default='result/020000000047540.txt', type=str)
    args = parser.parse_args()

    if args.model == 'cvae':
        test_cvae(args.file)


