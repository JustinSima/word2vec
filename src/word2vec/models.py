""" Implementation of CBOW and Skipgram models, with utilities for loading models and saving embeddings."""
import pathlib
from typing import Literal, Union

import numpy as np
import tensorflow as tf


class CBOW(tf.keras.Model):
    """ tf.keras.Model implemention Continuous Bag of Words model.
    
    Contains an embedding layer of size vocab_size x embedding_dim, which are our desired embeddings.
    Call 'save_embeddings' to save this layer as a numpy array.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding_layer = tf.Variable(tf.random.normal((vocab_size, embedding_dim)), trainable=True)
        self.output_layer = tf.Variable(tf.random.normal((embedding_dim, vocab_size)), trainable=True)
        
    def call(self, batch):
        embedding = tf.linalg.matmul(batch, self.embedding_layer)
        avg_embedding = tf.reduce_mean(embedding, axis=1, keepdims=False)

        linear_output = tf.linalg.matmul(avg_embedding, self.output_layer)
        output = tf.nn.softmax(linear_output)
        
        return output
    
    def save_embeddings(self, save_dir: Union[str, pathlib.Path]):
        if isinstance(save_dir, pathlib.Path):
            save_dir = pathlib.Path(save_dir)
        save_path = save_dir / 'embeddings.npy'
        print('SAVE PATH', save_path)

        embedding_tensor = tf.convert_to_tensor(self.embedding_layer)
        np.save(save_path, embedding_tensor.numpy())


class SkipGram(tf.keras.Model):
    """ tf.keras.Model implemention Skipgram model.
    
    Contains an embedding layer of size vocab_size x embedding_dim, which are our desired embeddings.
    Call 'save_embeddings' to save this layer as a numpy array.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding_layer = tf.Variable(tf.random.normal((vocab_size, embedding_dim)), trainable=True)
        self.output_layer = tf.Variable(tf.random.normal((embedding_dim, vocab_size)), trainable=True)
        self.embeddings = None
        
    def call(self, batch):
        embeddings = tf.linalg.matmul(batch, self.embedding_layer)
        linear_output = tf.linalg.matmul(embeddings, self.output_layer)
        output = tf.nn.softmax(linear_output)
        
        return output

    def save_embeddings(self, save_path: str):
        if not save_path.endswith('.npy'):
            save_path += '.npy'
        embedding_tensor = tf.convert_to_tensor(self.embedding_layer)
        np.save(save_path, embedding_tensor.numpy())

def load_model(model_type: Literal['cbow', 'skipgram'], embedding_dim: int, vocab_size: int) -> tf.keras.Model:
    """ Returns an untrained instantiation of a CBOW or Skipgram model class."""
    if model_type == 'cbow':
        return CBOW(vocab_size=vocab_size, embedding_dim=embedding_dim)
    
    elif model_type == 'skipgram':
        return SkipGram(vocab_size=vocab_size, embedding_dim=embedding_dim)
    
    else:
        raise ValueError(f"Parameter 'model_type' expected to be one of ('cbow', 'skipgram'). Received: {model_type}")
