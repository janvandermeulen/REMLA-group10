"""
phishing_detection_script.py

This script performs phishing detection using a machine learning model. It loads data,
preprocesses it, builds a model, trains the model, evaluates its performance, and plots
a confusion matrix.

Dependencies:
    - os
    - yaml
    - phishing_detection.train.train
    - phishing_detection.get_data.load_data
    - phishing_detection.model_definition.build_model
    - phishing_detection.predict.evaluate_results
    - phishing_detection.predict.plot_confusion_matrix
    - phishing_detection.predict.predict_classes
    - phishing_detection.preprocess.preprocess_data

Usage:
    Ensure that the parameters are specified in the 'params.yaml' file located in the
    'phishing-detection/phishing_detection' directory. Then run this script.

Example:
    python phishing_detection_script.py
"""

import os
import yaml
from phishing_detection.train import train
from phishing_detection.get_data import load_data
from phishing_detection.model_definition import build_model
from phishing_detection.predict import evaluate_results, plot_confusion_matrix, predict_classes
from phishing_detection.preprocess import preprocess_data



def run(params: dict) -> None:
    """
    Runs the model.

    Parameters:
        params (dict): A dictionary containing parameters for the model training and evaluation.

    Returns:
        None

    Example:
        params = yaml.safe_load(path)
        run(params)
    """
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, char_index = preprocess_data(
        X_train, y_train, X_val, y_val, X_test, y_test)

    # Create model
    model = build_model(char_index, params)

    # Train model
    train(model, X_train, y_train, X_val, y_val, params)

    # Evaluate model
    prediction = predict_classes(model, X_test)
    evaluation_results = evaluate_results(y_test, prediction)

    # plot confusion matrix
    plot_confusion_matrix(evaluation_results['confusion_matrix']) # TODO save fig?

if __name__ == "__main__":
    parameters = yaml.safe_load(
        open(os.path.join("phishing-detection", "phishing_detection", "params.yaml")))
    run(parameters)
