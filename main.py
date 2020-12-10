from load_data import load_dialog_data
from model import create_seq2seqmodel
from train import train

if __name__ == '__main__':
    train_file_path = 'dailydialog_src_tgt_train.tsv'
    val_file_path = 'dailydialog_src_tgt_val.tsv'
    test_file_path = 'dailydialog_src_tgt_test.tsv'
    dataset = load_dialog_data(train_file_path, val_file_path, test_file_path, batch_size=32)
    dataset_reader = dataset['dataset_reader']
    train_loader = dataset['train_loader']
    val_loader = dataset['val_loader']
    test_data = dataset['test_data']
    model = create_seq2seqmodel(vocab=dataset['vocab'])
    train(model, dataset_reader=dataset['dataset_reader'],
          train_loader=train_loader, val_loader=val_loader,
          test_data=test_data, num_epochs=30)
