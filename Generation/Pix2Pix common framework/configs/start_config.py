from trainers.default_trainer import DefaultTrainerConfig
from models.generator_u_net import ModifiedUNetConfig
from data_loaders.load_cmp_facade import DataLoaderConfig

trainer_config = DefaultTrainerConfig(
    experiment_name="building_generate",
    _lambda=100,
    generator_learning_rate=2e-4,
    discriminator_learning_rate=2e-4,
    epoch=100,
    log_root="logs",
    save_freq=5,
    predict_dpi=300
)

generator_config = ModifiedUNetConfig(
    output_channel=3
)

data_loader_config = DataLoaderConfig(
    resize_up_size=300,
    output_size=256,
    buffer_size=400,
    batch_size=1,
    test_batch_size=3
)
