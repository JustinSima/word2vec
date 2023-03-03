""" Load dataset, training embedding model, and save embeddings.

Training parameters are imported from parameters.py.
"""
import tensorflow as tf

import models
import preprocessing

# Import training parameters.
from parameters import *


# -- Load training data and initialize model.
if not DATA_DIR.exists():
    DATA_DIR.mkdir()

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()

X, y, vocab_size = preprocessing.load_and_process_dataset(model_type=MODEL_TYPE, dataset_name=DATASET_NAME, window_size=WINDOW_SIZE, vocab_dir=OUTPUT_DIR)
model = models.load_model(model_type=MODEL_TYPE, embedding_dim=EMBEDDING_DIM, vocab_size=vocab_size)

# -- Train model.
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

print('Fitting model...', end=' ')

model.fit(
    x=X,
    y=y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE
)
print('Done.')

# -- Save learned embeddings.
model.save_embeddings(OUTPUT_DIR)
print('Embeddings saved.')
