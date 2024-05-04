from keras._tf_keras.keras.models import Sequential, Model
from keras._tf_keras.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout


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
