from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp_models.generation import Seq2SeqPredictor
import itertools


def train(model, dataset_reader, train_loader,
          val_loader=None, test_data=None, num_epochs=1):
    optimizer = AdamOptimizer(model.named_parameters())
    trainer = GradientDescentTrainer(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     validation_data_loader=val_loader,
                                     num_epochs=num_epochs)
    for i in range(10):
        print('Epoch: {}'.format(i * num_epochs))
        trainer.train()
        predictor = Seq2SeqPredictor(model, dataset_reader)

        for instance in itertools.islice(test_data, 10):
            print('SOURCE:', instance.fields['source_tokens'].tokens)
            print('GOLD:', instance.fields['target_tokens'].tokens)
            print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])
            print('-' * 20)
