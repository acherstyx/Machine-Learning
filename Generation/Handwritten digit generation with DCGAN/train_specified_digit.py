from configs.specified_digit import *
from data_loaders.load_mnist import LoadMNIST
from models.specified_digits.dcgan_discriminator import DCGANDiscriminator
from models.specified_digits.dcgan_generator import DCGANGenerator
from trainers.specified_digit import Trainer


if "__main__" == __name__:
    dataset = LoadMNIST(DataLoaderConfig).get_dataset()

    trainer = Trainer(
        generator=DCGANGenerator(GeneratorConfig).get_model(),
        discriminator=DCGANDiscriminator().get_model(),
        data=LoadMNIST(DataLoaderConfig).get_dataset(),
        config=TrainerConfig
    )

    trainer.train()
