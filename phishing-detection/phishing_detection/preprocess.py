from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

def preprocess_data(raw_X_train: list[str], raw_y_train: list[str], raw_X_val: list[str], raw_y_val: list[str], raw_X_test: list[str], raw_y_test: list[str], sequence_length: int = 200): 
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