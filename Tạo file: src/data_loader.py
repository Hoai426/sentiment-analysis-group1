import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_imdb_data(num_words=10000, maxlen=500):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=num_words
    )
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    return x_train, y_train, x_test, y_test

def get_word_index():
    word_index = tf.keras.datasets.imdb.get_word_index()
    word_index = {k: (v+3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    reverse_word_index = {v: k for k, v in word_index.items()}
    return word_index, reverse_word_index
