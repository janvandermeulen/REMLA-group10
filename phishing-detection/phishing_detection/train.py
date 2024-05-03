from keras import Model
from keras.callbacks import History
import numpy as np

def train(model: Model, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, params: dict) -> History:
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


    hist = model.fit(X_train, y_train,
                    batch_size=params['batch_train'],
                    epochs=params['epoch'],
                    shuffle=True,
                    validation_data=(X_val, y_val)
                    )
    
    return hist