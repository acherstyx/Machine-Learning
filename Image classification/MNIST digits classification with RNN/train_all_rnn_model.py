import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.all_rnn import AllRNN
from data_loaders.load_mnist import LoadMNIST
from trainers.mnist_trainer import MNISTTrainer
from configs.all_rnn import *

if __name__ == "__main__":
    data_loader = LoadMNIST(load_mnist_config).get_dataset()
    model = AllRNN(row_by_row_rnn_config).show_summary(with_plot=True).get_model()
    trainer = MNISTTrainer(model, data_loader, mnist_trainer_config)

    trainer.train()
