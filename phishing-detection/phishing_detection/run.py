"""
This script performs phishing detection using a machine learning model. It loads data,
preprocesses it, builds a model, trains the model, evaluates its performance, and plots
a confusion matrix.

Usage:
    Ensure that the parameters are specified in the 'params.yaml' file located in the
    'phishing-detection/phishing_detection' directory. Then run this script.

Example:
    python phishing-detection/phishing_detection/run.py
"""

import os
import yaml
from phishing_detection.train import train
from phishing_detection.get_data import load_data
from phishing_detection.model_definition import build_model
from phishing_detection.predict import evaluate_results, plot_confusion_matrix, predict_classes
from phishing_detection.preprocess import preprocess_data




def run(params: dict, paramspath) -> None:
    """
    Runs the model with the given parameters.

    Parameters:
        params (dict): A dictionary containing parameters for the model training and evaluation.
        path (String): Path to parammeter file

    Returns:
        None

    Example:
        params = yaml.safe_load(path)
        run(params)
    """
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(paramspath)

    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, char_index = preprocess_data(
        X_train, y_train, X_val, y_val, X_test, y_test)

    # Create model
    model = build_model(char_index, params)

    # Train model
    model = train(model, X_train, y_train, X_val, y_val, params)

    # Evaluate model
    prediction = predict_classes(model, X_test)
    evaluation_results = evaluate_results(y_test, prediction)

    # plot confusion matrix
    plot_confusion_matrix(evaluation_results['confusion_matrix']) #save fig?

if __name__ == "__main__":
    path = os.path.join("phishing-detection", "phishing_detection", "params.yaml")
    with open(path ,encoding="UTF-8" ) as file:
        parameters = yaml.safe_load(file)
    run(parameters, path)
