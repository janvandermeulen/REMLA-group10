"""
Provides functions for training a model.

"""
import sys
import os
import yaml
import numpy as np

from keras._tf_keras.keras import Model
from keras._tf_keras.keras.models import load_model

def train(model: Model, X_train: np.array, y_train: np.array,
          X_val: np.array, y_val: np.array, params: dict) -> Model:
    """
    Train the model.

    Args:
        model: A Keras model.
        x_train: The training data.
        y_train: The training labels.
        x_val: The validation data.
        y_val: The validation labels.
        params: A dictionary containing the parameters for the model.

    Returns:
        The history object returned by model.fit().

    """
    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])


    model.fit(X_train, y_train,
                batch_size=params['batch_train'],
                epochs=params['epoch'],
                shuffle=True,
                validation_data=(X_val, y_val)
                )

    return model

def main():
    """
    Train and save model.

    Returns:
        None
    """
    path = sys.argv[1]

    X_train = np.load(f"{path}/preprocess/X_train.npy")
    y_train = np.load(f"{path}/preprocess/y_train.npy")
    X_val = np.load(f"{path}/preprocess/X_val.npy")
    y_val = np.load(f"{path}/preprocess/y_val.npy")
    parampath = os.path.join("phishing-detection", "phishing_detection", "params.yaml")
    with open(parampath ,encoding="UTF-8" ) as file:
        params = yaml.safe_load(file)
    model = load_model(f"{path}/model/initial_model.keras")

    trained_model = train(model, X_train, y_train, X_val, y_val, params)

    trained_model.save(f"{path}/model/trained_model.keras")

if __name__ == "__main__":
    main()
