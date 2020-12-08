if __name__ == '__main__':
    vocab_file = open('data/movie_25000')
    data = open('data/s_given_t_dialogue_length2_6.txt', encoding='utf-8')
from allennlp_models.generation import Seq2SeqDatasetReader