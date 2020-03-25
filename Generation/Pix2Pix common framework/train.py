from data_loaders.load_cmp_facade import DataLoader
from models.generator_u_net import ModifiedUNet
from models.discriminator_patch_gan import PatchGAN
from trainers.default_trainer import DefaultTrainer

from configs.start_config import trainer_config, data_loader_config, generator_config

if __name__ == "__main__":
    data_loaders = DataLoader(data_loader_config)
    data_loaders.load()

    generator = ModifiedUNet(generator_config).build().show_summary(with_plot=True).get_model()
    discriminator = PatchGAN(None).build().show_summary(with_plot=True).get_model()

    trainer = DefaultTrainer(generator,
                             discriminator,
                             data_loaders.get_dataset(),
                             trainer_config)
    trainer.train()
