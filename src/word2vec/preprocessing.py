""" Functions for loading a HuggingFace text dataset as a training set for either CBOW or Skipgram model."""
import json
import pathlib
import re
from typing import Literal, Union

import datasets
import tensorflow as tf

from parameters import DATA_DIR


def one_hot(index: int, vocab_size: int) -> tf.Tensor:
    """ Returns one-hot encoded tensor for given vocabulary index and size."""
    ohe = [0. for _ in range(vocab_size)]
    ohe[index] = 1.
    
    return tf.convert_to_tensor(ohe, dtype=tf.float32)

def load_and_process_dataset(
    model_type: Literal['cbow', 'skipgram'],
    dataset_name: str,
    window_size: int,
    vocab_dir: Union[str, pathlib.Path]
) -> tuple[tf.Tensor, tf.Tensor, int]:
    """ Load HF dataset in format expected for training.
    Returns features, labels, and vocab size.
    """
    # Ensure a valid model is specified.
    if model_type not in ('cbow', 'skipgram'):
        raise ValueError(f"Parameter 'model_type' expected to be one of ('cbow', 'skipgram'). Received: {model_type}")
    
    print('Preparing training data...', end=' ')

    # Load HF dataset.
    ds = datasets.load_dataset(
        dataset_name,
        cache_dir=DATA_DIR / dataset_name,
        split='train',
        streaming=False
    )
    text = ds['text'][0].lower()

    # Covvert to list of words and create vocabulary.
    filtered_text = " ".join(re.findall("[a-zA-Z0-9]+", text))
    words = filtered_text.split()
    vocab = sorted(list(set(words)))
    
    vocab_size = len(vocab)
    index_mapping = {i:s for i, s in enumerate(vocab)}
    string_mapping = {s:i for i, s in index_mapping.items()}

    # Prepare contexts.
    context_size = 2*window_size + 1
    contexts = []
    for sample in zip(*[words[i:] for i in range(context_size)]):
        idxs = [string_mapping[t] for t in sample]
        contexts.append(idxs)

    # Prepare training dataset according to model type.
    if model_type == 'cbow':
        # Collate cbow training data.
        X = [cont[:window_size] + cont[window_size+1:] for cont in contexts]
        y = [cont[window_size] for cont in contexts]
        X = tf.convert_to_tensor([[one_hot(idx, vocab_size) for idx in sample] for sample in X], dtype=tf.float32)
        y = tf.convert_to_tensor([one_hot(idx, vocab_size) for idx in y], dtype=tf.float32)
    
    else:
        # Collate skipgram training data.
        X = []
        y = []
        for cont in contexts:
            context_idxs = cont[:window_size] + cont[window_size+1:]
            center_idx = cont[window_size]
            
            X = [*X, *context_idxs]
            y = [*y, *[center_idx for _ in range(context_size-1)]]
        
        X = tf.convert_to_tensor([one_hot(idx, vocab_size) for idx in X])
        y = tf.convert_to_tensor([one_hot(idx, vocab_size) for idx in y])
        
    # Save vocabulary mappings.
    if not isinstance(vocab_dir, pathlib.Path):
        vocab_dir = pathlib.Path(vocab_dir)
        
    with open(vocab_dir / 'string-to-index.json', 'w') as f:
        json.dump(string_mapping, f)

    with open(vocab_dir / 'index-to-string.json', 'w') as f:
        json.dump(index_mapping, f)
        
    print('Done.')
    
    return X, y, vocab_size
