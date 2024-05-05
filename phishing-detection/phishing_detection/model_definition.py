"""
Provides functions to create the model.

"""
from keras._tf_keras.keras.models import Sequential, Model
from keras._tf_keras.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import yaml
import sys
import numpy as np
import os
import utils
def build_model(char_index: dict, params: dict) -> Model:
    """
    Build a model for the phishing detection task
    
    Args:
        char_index: A dictionary mapping characters to their index.
        params: A dictionary containing the parameters for the model.
    
    Returns:
        A Keras model.

    """
    voc_size = len(char_index.keys())
    #print("voc_size: {}".format(voc_size))  # TODO remove if not needed for anything, if needed use fstring instead
    dropout_rate = 0.2  # This can be parameterized in `params` if varying dropout rates are needed.

    model = Sequential()
    model.add(Embedding(voc_size + 1, 50))

    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())

    model.add(Dense(len(params['categories'])-1, activation='sigmoid'))

    return model

def main():
    path = sys.argv[1]
    params = yaml.safe_load(open(os.path.join("phishing-detection", "phishing_detection", "params.yaml")))
    char_index = utils.load_json(f"{path}/preprocess/char_index.json")

    model = build_model(char_index, params)
    model.save(f"{path}/model/initial_model.keras")

if __name__ == "__main__":
    main()
