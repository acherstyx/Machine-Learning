from models.rnn import RowByRowRNNConfig
from trainers.mnist_trainer import MNISTTrainerConfig
from data_loaders.load_mnist import LoadMNISTConfig


# learning rate scheduler
def scheduler(epoch):
    if epoch <= 10:
        return 1e-3
    elif epoch <= 15:
        return 5e-5
    else:
        return 1e-5


# trainer
mnist_trainer_config = MNISTTrainerConfig(
    experiment_name="MNIST_all_RNN",
    log_root="logs",
    epochs=20,
    scheduler=scheduler,
    clip_norm=10.0
)

# model
row_by_row_rnn_config = RowByRowRNNConfig(
    image_size=28,
    unit_stack=[64, 128, 256, 128, 10],
    dropout_rate=0.2
)

# data loader
load_mnist_config = LoadMNISTConfig(
    batch_size=10,
    drop_reminder=True
)
