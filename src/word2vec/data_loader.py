""" Creates TensorFlow datasets for model training."""
import tensorflow as tf
import tensorflow_datasets as tfds
from definitions import (
    VOCAB_SIZE,
    SEQUENCE_LENGTH,
    DATA_DIR
)


def get_tf_dataset(split: str=None):
    data_dir = DATA_DIR / 'wiki40b'
    dataset = tfds.load(
        name='wiki40b/en',
        split=split,
        data_dir=data_dir,
        download=True
    )
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=' ',
        oov_token='<unk>'
    )

    return dataset

def prepare_skipgram_batch():
    pass

def prepare_cbow_batch():
    pass


dataset = get_tf_dataset(split='train')
for thing in dataset:
    print(thing)

    break
