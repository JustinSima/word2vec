""" Defines parameters for model, training, dataset caching, and output location."""
import pathlib


# Directory for dataset caching.
DATA_DIR = pathlib.Path(__file__).parent.parent.parent / 'data'

# Directory for storing results.
OUTPUT_DIR = DATA_DIR / 'outputs'

# Model parameters.
DATASET_NAME = 'tiny_shakespeare'
MODEL_TYPE = 'cbow' # 'skipgram'
EMBEDDING_DIM= 16
WINDOW_SIZE = 2

# Training parameters.
EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
VERBOSE = 3
