import json
from sklearn.model_selection import train_test_split
from random import choice, shuffle, seed
import random
import re
from collections import Counter
from tqdm import tqdm


def discriminator_dataset(file_path):
    turns = []
    pattern = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

    with open(file_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            turn = line['turns']
            turn = [re.sub(pattern, ' ', x.strip().replace('"', " ")) for x in turn]
            turns.append(turn)
            # print(len(turn), end=' ')
    train, val_test = train_test_split(turns, test_size=0.2, random_state=20020206)
    val, test = train_test_split(val_test, test_size=0.5, random_state=20020206)
    dataset = [train, val, test]
    for name, data in zip(['train', 'valid', 'test'], dataset):
        num_same_person = 0
        num_same_topic = 0
        num_diff_person = 0
        with open(name + '_disc.tsv', 'w', encoding='utf-8') as f:
            for turn in tqdm(data):
                for i in range(len(turn)):
                    utt1 = turn[i]
                    if i + 4 >= len(turn):
                        continue
                    utt2 = turn[i + 4]
                    f.write(utt1)
                    f.write('\t')
                    f.write(utt2)
                    f.write('\t')
                    f.write('1\t')  # 同一topic
                    num_same_topic += 1
                    f.write('1\n')
                    num_same_person += 1

                    if i + 5 >= len(turn):
                        continue
                    utt2 = turn[i + 5]
                    f.write(utt1)
                    f.write('\t')
                    f.write(utt2)
                    f.write('\t')
                    f.write('1\t')  # 同一topic
                    num_same_topic += 1
                    f.write('0\n')
                    num_diff_person += 1
            shuffle(data)
            num_neg_samples = num_same_topic
            for k in tqdm(range(num_neg_samples)):
                idx = random.randrange(0, num_neg_samples)
                utt1 = choice(data[idx % len(data)])
                idx = random.randrange(0, num_neg_samples)
                utt2 = choice(data[idx % len(data)])
                f.write(utt1)
                f.write('\t')
                f.write(utt2)
                f.write('\t')
                f.write('0\t')  # 不同topic
                f.write('0\n')  # 不同person


def generator_dataset(file_path):
    turns = []
    pattern = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    with open(file_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            turn = line['turns']
            turn = [re.sub(pattern, ' ', x.strip().replace('"', " ")) for x in turn]
            turns.append(turn)
    train, val_test = train_test_split(turns, test_size=0.2, random_state=20020206)
    val, test = train_test_split(val_test, test_size=0.5, random_state=20020206)
    dataset = [train, val, test]
    for name, data in zip(['train', 'valid', 'test'], dataset):
        with open(name + '_gen.tsv', 'w', encoding='utf-8') as f:
            for turn in tqdm(data):
                for i in range(len(turn)):
                    utt1 = turn[i]
                    if i + 1 < len(turn):
                        utt2 = turn[i + 1]
                        f.write(utt1)
                        f.write('\t')
                        f.write(utt2)
                        f.write('\t')
                        f.write('1\t0\n')  # 同一topic 不同人


if __name__ == '__main__':
    seed(20020206)
    discriminator_dataset('all.txt')
    generator_dataset('all.txt')
    from time import sleep
    # generate_for_disc('all.txt')
    # with open('metalwoz-v1/train.tsv', 'r') as f:
    #     for line in f:
    #         line = line.split('\t')
    #         line[-2] = int(line[-2])
    #         line[-1] = int(line[-1].strip('\n'))
    #         # print(line)
    #         # sleep(0.1)
    #         try:
    #             assert "\"" not in line[0]
    #             assert "\"" not in line[1]
    #             assert len(line) == 4
    #             assert isinstance(line[0], str)
    #             assert isinstance(line[1], str)
    #             assert isinstance(line[-1], int)
    #             assert isinstance(line[-2], int)
    #             assert line[-1] == 0 or line[-1] == 1
    #             assert line[-2] == 0 or line[-2] == 1
    #         except:
    #             print(line)


    # pattern = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    # text = 'utt1#$^^}{":<>?~"""??(+??@splits@utt2??@splits@utt3@splits@utt4(*)&*)%$^&@splits@'
    # text = re.sub(pattern, ' ', text)
    # print(text.split('@splits@'))

