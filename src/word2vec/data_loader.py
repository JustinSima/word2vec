""" Creates TensorFlow datasets for model training."""
import pathlib
from typing import Union
import tensorflow as tf
import datasets

from definitions import (
    VOCAB_SIZE,
    SEQUENCE_LENGTH,
    DATA_DIR,
    BATCH_SIZE
)


def get_tf_dataset(dataset_name: Union[str, pathlib.Path]='bookcorpus', split: str=None):
    data_dir = DATA_DIR / dataset_name
    dataset = datasets.load_dataset(
        dataset_name,
        cache_dir=data_dir,
        split=split,
        streaming=False
    )
    tf_dataset = dataset.to_tf_dataset(
        columns=None,
        shuffle=False,
        batch_size=BATCH_SIZE,
        collate_fn=None
    )
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=' ',
        oov_token='<unk>'
    )

    return dataset

def collate_skipgram_batch():
    pass

def collate_cbow_batch():
    pass

# Example using shakespeare since I don't have reviews handy.
dataset = get_tf_dataset('bookcorpus', split='train')
for batch in dataset:
    print(batch)

    break
