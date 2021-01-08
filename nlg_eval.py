from nlgeval import NLGEval
import argparse
from collections import defaultdict
from tqdm import tqdm
import os
import json
from metrics import distinct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypothesis', default=None, type=str)
    parser.add_argument('--references', default=None, type=str)
    args = parser.parse_args()
    hyps = []
    refs = []
    with open(file=args.hypothesis, mode='r', encoding='utf-8') as fr:
        for line in fr.readlines():
            hyps.append(line)
    with open(file=args.references, mode='r', encoding='utf-8') as fr:
        for line in fr.readlines():
            refs.append(line)
    assert len(hyps) == len(refs)
    nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)
    metrics_dict = nlgeval.compute_metrics([refs], hyps)
    hyps_inter_dist1, hyps_inter_dist2 = distinct(hyps)
    refs_inter_dist1, refs_inter_dist2 = distinct(refs)
    metrics_dict['hyps_inter_dist1'] = hyps_inter_dist1
    metrics_dict['hyps_inter_dist2'] = hyps_inter_dist2
    metrics_dict['refs_inter_dist1'] = refs_inter_dist1
    metrics_dict['refs_inter_dist2'] = refs_inter_dist2
    '''
    metrics_dict = defaultdict(float)
    metrics_dict['hypothesis'] = args.hypothesis
    metrics_dict['references'] = args.references
    for ref, hyp in zip(tqdm(refs), hyps):
        cur_metrics_dict = nlgeval.compute_individual_metrics([ref], hyp)
        for key, value in cur_metrics_dict.items():
            metrics_dict[key] += value
    for key, value in metrics_dict.items():
        metrics_dict[key] = value / len(hyps)
    '''
    print(metrics_dict)
    save_dir = os.path.split(args.hypothesis)[0]
    with open(os.path.join(save_dir, 'nlgeval_metrics.txt'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)
