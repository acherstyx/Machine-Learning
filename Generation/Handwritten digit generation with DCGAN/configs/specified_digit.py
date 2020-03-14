class DataLoaderConfig:
    BATCH_SIZE = 64
    DROP_REMINDER = True


class GeneratorConfig:
    NOISE_DIM = 2


class TrainerConfig:
    EXPERIMENT_NAME = "generate_specified_number"
    EPOCH = 30
    RANDOM_SEED = 9999

    LOG_ROOT = "logs"

    CHECKPOINT_PATH = "checkpoint"
    GENERATE_PATH = "generate_output"

    GIF_NAME = "generate_history.gif"

    GENERATOR_LEARNING_RATE = 5e-5
    DISCRIMINATOR_LEARNING_RATE = 5e-5

    TOTAL_IMAGES = 70000

    CHECKPOINT_SAVE_FREQUENCY = 1

    # reuse
    BATCH_SIZE = DataLoaderConfig.BATCH_SIZE
    NOISE_DIM = GeneratorConfig.NOISE_DIM
