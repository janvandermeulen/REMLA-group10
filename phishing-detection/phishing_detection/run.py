from phishing_detection import train
from phishing_detection.get_data import load_data
from phishing_detection.model_definition import build_model
from phishing_detection.predict import evaluate_results, plot_confusion_matrix, predict_classes
from phishing_detection.preprocess import preprocess_data


def run(params):
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, char_index = load_data()
    
    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, char_index = preprocess_data(X_train, y_train, X_val, y_val, X_test, y_test)

    # Create model
    model = build_model(params)

    # Train model
    hist = train(model, X_train, y_train, X_val, y_val, params)

    # Evaluate model
    prediction = predict_classes(model, X_test)
    evaluation_results = evaluate_results(y_test, prediction)

    # plot confusion matrix
    plot_confusion_matrix(evaluation_results['confusion_matrix'])
    