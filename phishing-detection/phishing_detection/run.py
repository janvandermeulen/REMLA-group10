import os
import yaml
from phishing_detection.train import train
from phishing_detection.get_data import load_data
from phishing_detection.model_definition import build_model
from phishing_detection.predict import evaluate_results, plot_confusion_matrix, predict_classes
from phishing_detection.preprocess import preprocess_data



def run(params: dict) -> None:
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, char_index = preprocess_data(
        X_train, y_train, X_val, y_val, X_test, y_test)

    # Create model
    model = build_model(char_index, params)

    # Train model
    hist = train(model, X_train, y_train, X_val, y_val, params)

    # Evaluate model
    prediction = predict_classes(model, X_test)
    evaluation_results = evaluate_results(y_test, prediction)

    # plot confusion matrix
    fig = plot_confusion_matrix(evaluation_results['confusion_matrix']) # TODO save fig?

if __name__ == "__main__":
    parameters = yaml.safe_load(
        open(os.path.join("phishing-detection", "phishing_detection", "params.yaml")))
    run(parameters)
