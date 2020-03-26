from models.row_by_row_rnn import RowByRowRNNConfig
from trainers.mnist_trainer import MNISTTrainerConfig
from data_loaders.load_mnist import LoadMNISTConfig

row_by_row_rnn_config = RowByRowRNNConfig(
    image_size=28,
    unit_stack=[64, 128, 256, 1024, 2048]
)


load_mnist_config = LoadMNISTConfig(
    batch_size=10,
    drop_reminder=True
)

mnist_trainer_config = MNISTTrainerConfig(
    epochs=30,
    learning_rate=1e-3
)
