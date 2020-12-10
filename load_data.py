from allennlp_models.generation import Seq2SeqDatasetReader
from allennlp.data.tokenizers import WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import PyTorchDataLoader as DataLoader


def load_dialog_data(train_file_path, val_file_path=None, test_file_path=None,
                     tokenizer=None, token_indexer=None, delimiter='\t', batch_size=32):
    tokenizer = tokenizer if tokenizer else WhitespaceTokenizer()
    token_indexers = {'tokens': token_indexer if token_indexer else SingleIdTokenIndexer()}
    # tokenizer = PretrainedTransformerTokenizer('facebook/bart-base')
    # token_indexers = {'tokens': PretrainedTransformerIndexer('facebook/bart-base', namespace="tokens")}

    dataset_reader = Seq2SeqDatasetReader(source_tokenizer=tokenizer,
                                          target_tokenizer=tokenizer,
                                          source_token_indexers=token_indexers,
                                          target_token_indexers=token_indexers,
                                          delimiter=delimiter,
                                          # source_add_end_token=False,
                                          # source_add_start_token=False,
                                          # target_add_end_token=False,
                                          # target_add_start_token=False
                                          )
    train_data = dataset_reader.read(file_path=train_file_path)
    # train_data.instances = train_data.instances[:1000] # test用
    val_data = None
    test_data = None
    val_loader = None
    test_loader = None

    vocab = Vocabulary.from_instances(train_data)
    train_data.index_with(vocab)

    if val_file_path:
        val_data = dataset_reader.read(file_path=val_file_path)
        val_data.index_with(vocab)
        val_loader = DataLoader(val_data, batch_size=batch_size)

    if test_file_path:
        test_data = dataset_reader.read(file_path=test_file_path)
        test_data.index_with(vocab)
        test_loader = DataLoader(test_data, batch_size=batch_size)

    batch_sampler = BucketBatchSampler(train_data, batch_size=batch_size, sorting_keys=["source_tokens", "target_tokens"],
                                       padding_noise=0)
    train_loader = DataLoader(train_data, batch_sampler=batch_sampler)

    return {'dataset_reader': dataset_reader, 'vocab': vocab, 'train_data': train_data,
            'val_data': val_data, 'test_data': test_data, 'train_loader': train_loader, 
            'val_loader': val_loader, 'test_loader': test_loader}
