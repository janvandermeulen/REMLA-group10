"""
Provides functions to preprocess data.

"""
import sys
import utils
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import classification_report, confusion_matrix
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
#from keras._tf_keras.keras.models import Sequential
#from keras._tf_keras.keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten


def preprocess_data(raw_X_train: list[str], raw_y_train: list[str],
                    raw_X_val: list[str], raw_y_val: list[str],
                    raw_X_test: list[str], raw_y_test: list[str], sequence_length: int = 200
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """
    Preprocess the data for training the model.

    Args:
        raw_X_train: List of strings containing the training data.
        raw_y_train: List of strings containing the training labels.
        raw_X_val: List of strings containing the validation data.
        raw_y_val: List of strings containing the validation labels.
        raw_X_test: List of strings containing the test data.
        raw_y_test: List of strings containing the test labels.
        sequence_length: The length of the sequences to pad the data to.

    Returns:
        Tuple of preprocessed data.

    """

    # Tokenize the dataset
    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_X_train + raw_X_val + raw_X_test)
    char_index = tokenizer.word_index

    X_train = pad_sequences(tokenizer.texts_to_sequences(raw_X_train), maxlen=sequence_length)
    X_val = pad_sequences(tokenizer.texts_to_sequences(raw_X_val), maxlen=sequence_length)
    X_test = pad_sequences(tokenizer.texts_to_sequences(raw_X_test), maxlen=sequence_length)
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, char_index

def main():
    """
    Preprocess data and save result to file.

    Returns:
        None
    """
    path = sys.argv[1]

    # Load data from text files
    raw_X_train = utils.load_data_from_text(f"{path}/raw/X_train.txt")
    raw_y_train = utils.load_data_from_text(f"{path}/raw/y_train.txt")
    raw_X_val = utils.load_data_from_text(f"{path}/raw/X_val.txt")
    raw_y_val = utils.load_data_from_text(f"{path}/raw/y_val.txt")
    raw_X_test = utils.load_data_from_text(f"{path}/raw/X_test.txt")
    raw_y_test = utils.load_data_from_text(f"{path}/raw/y_test.txt")


    X_train, y_train, X_val, y_val, X_test, y_test, char_index = preprocess_data(
        raw_X_train, raw_y_train, raw_X_val, raw_y_val, raw_X_test, raw_y_test)

    np.save(f"{path}/preprocess/X_train.npy", X_train)
    np.save(f"{path}/preprocess/y_train.npy", y_train)
    np.save(f"{path}/preprocess/X_val.npy", X_val)
    np.save(f"{path}/preprocess/y_val.npy", y_val)
    np.save(f"{path}/preprocess/X_test.npy", X_test)
    np.save(f"{path}/preprocess/y_test.npy", y_test)
    utils.save_json(char_index, f"{path}/preprocess/char_index.json")


if __name__ == "__main__":
    main()
