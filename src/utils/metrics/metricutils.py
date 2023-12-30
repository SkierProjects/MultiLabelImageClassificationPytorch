from sklearn.metrics import f1_score as sklearnf1, precision_score, recall_score
import numpy as np

def f1_score(targets, predictions, average='micro'):
    """
    Compute F1 score for binary predictions.

    Parameters:
    - targets: array-like, true binary labels
    - predictions: array-like, binary predictions
    - average: string, [None, 'micro' (default), 'macro', 'samples', 'weighted']

    Returns:
    - f1: float, computed F1 score
    """
    return sklearnf1(targets, predictions, average=average, zero_division=1)

def compute_metrics(targets, outputs, threshold=0.5, average='micro'):
    """
    Compute precision, recall, and F1 score for each class.

    Parameters:
    - targets: array-like, true binary labels
    - outputs: array-like, raw output scores from the classifier
    - threshold: float, threshold for converting raw scores to binary predictions
    - average: string, [None (default), 'micro', 'macro', 'samples', 'weighted']
               This parameter is required for multilabel targets.

    Returns:
    - precision: float, precision score per class
    - recall: float, recall score per class
    - f1: float, F1 score per class
    """
    
    precision = precision_score(targets, outputs, average=average)
    recall = recall_score(targets, outputs, average=average)
    f1 = f1_score(targets, outputs, average=average)
    return precision, recall, f1

def f1_score_rawoutputs(targets, outputs, threshold=0.5, average='micro'):
    """
    Compute F1 score from raw outputs using a threshold.

    Parameters:
    - targets: array-like, true binary labels
    - outputs: tensor, raw output scores from the classifier
    - threshold: float, threshold for converting raw scores to binary predictions
    - average: string, [None, 'micro' (default), 'macro', 'samples', 'weighted']

    Returns:
    - f1: float, computed F1 score
    """
    predictions = getpredictions_with_threshold(outputs, threshold)
    f1 = f1_score(targets, predictions, average=average)
    return f1

def getpredictions_with_threshold(outputs, threshold=0.5):
    """
    Convert raw output scores to binary predictions using a threshold, using numpy arrays.

    Parameters:
    - outputs: numpy.ndarray, raw output scores from the classifier
    - threshold: float, threshold for converting raw scores to binary predictions

    Returns:
    - predictions: numpy.ndarray, binary predictions
    """
    if threshold == None:
        threshold = 0.5
    # Apply sigmoid to the outputs to get probabilities
    probabilities = 1 / (1 + np.exp(-outputs))  # Sigmoid function
    # Apply threshold to the probabilities to get binary predictions
    predictions = (probabilities > threshold).astype(int)  # Convert to integer type
    return predictions