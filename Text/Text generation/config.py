import os

# dataset
WINDOWS_SIZE = 100
SHIFT = 1
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000

# model
EMBEDDING_DIM = 256
RNN_UNITES = 1024

# save
CHECKPOINT_ROOT = ".log"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, "checkpoint_{epoch}")

# train
EPOCHS = 10

# generate
NUM_GENERATE = 1000