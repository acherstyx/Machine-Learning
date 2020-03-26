from models.row_by_row_rnn import RowByRowRNN
from data_loaders.load_mnist import LoadMNIST
from trainers.mnist_trainer import MNISTTrainer

from configs.defaults import *

if __name__ == "__main__":
    data_loader = LoadMNIST(load_mnist_config).get_dataset()
    model = RowByRowRNN(row_by_row_rnn_config).get_model()
    trainer = MNISTTrainer(model, data_loader, mnist_trainer_config)

    trainer.train()
