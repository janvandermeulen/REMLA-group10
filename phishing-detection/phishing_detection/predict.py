"""
Provides functions to peform predictions.

"""
import sys
import numpy as np
import seaborn as sns
import utils
from keras import Model
from keras._tf_keras.keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
def predict_classes(model: Model, x_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Predict class labels for samples in x_test.

    Args:
        model: Trained model to use for prediction.
        x_test: Test data.
        threshold: Threshold for converting probabilities to binary labels.

    Returns:
        Predicted binary labels for the samples in x_test.
    """
    y_pred = model.predict(x_test, batch_size=1000)

    # Convert predicted probabilities to binary labels
    y_pred_binary = (np.array(y_pred) > threshold).astype(int)

    return y_pred_binary


def evaluate_results(y_test: np.ndarray, y_pred_binary: np.ndarray) -> dict:
    """
    Evaluate the results of a binary classification task. 
    This function prints the classification report, confusion matrix, and accuracy score.

    Args:
        y_test: True binary labels.
        y_pred_binary: Predicted binary labels.

    Returns:
        A dictionary containing the classification report, confusion matrix, and accuracy score.
    """

    y_test=y_test.reshape(-1,1)

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    print('Classification Report:')
    print(report)

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:', confusion_mat)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_binary)
    print('Accuracy:', accuracy)

    return {'classification_report': report, 'confusion_matrix': confusion_mat,
            'accuracy': accuracy}


def plot_confusion_matrix(confusion_mat: np.ndarray) -> plt.Figure:
    """
    Plot a heatmap of the confusion matrix.

    Args:
        confusion_mat (array): Confusion matrix to plot.

    Returns:
        A matplotlib figure of the confusion matrix heatmap.

    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_mat, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    return plt.gcf()


def main():
    """
    Evaluate model and save plots

    Returns:
        None
    """
    path = sys.argv[1]
    # Load data from npy files
    X_test = np.load(f"{path}/preprocess/X_test.npy")
    y_test = np.load(f"{path}/preprocess/y_test.npy")
    model = load_model(f"{path}/model/trained_model.keras")

    prediction = predict_classes(model, X_test)
    evaluation_results = evaluate_results(y_test, prediction)
    utils.save_data_as_text(evaluation_results, f"{path}/results/results.txt")

    fig = plot_confusion_matrix(evaluation_results['confusion_matrix'])
    fig.savefig(f"{path}/results/confusion_matrix.pdf")


if __name__ == "__main__":
    main()
