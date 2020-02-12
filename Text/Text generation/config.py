import os

# dataset
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000

CHECKPOINT_ROOT = ".log"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_ROOT, "checkpoint_{epoch}")

EPOCHS = 10
