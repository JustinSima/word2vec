""" Defines constants for use throughout application."""
import pathlib
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE
VOCAB_SIZE = 4096
SEQUENCE_LENGTH = 10
DATA_DIR = pathlib.Path('../../data')
