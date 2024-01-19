from sklearn.metrics import f1_score as sklearnf1, precision_score, recall_score
import numpy as np
import torch
from imclaslib.logging.loggerfactory import LoggerFactory
from scipy.special import expit
logger = LoggerFactory.get_logger(f"logger.{__name__}")

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

def compute_metrics(targets, outputs, average='micro'):
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
    
    precision = precision_score(targets, outputs, average=average, zero_division=1)
    recall = recall_score(targets, outputs, average=average, zero_division=1)
    f1 = sklearnf1(targets, outputs, average=average, zero_division=1)
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

    # Apply sigmoid to the outputs to get probabilities
    probabilities = expit(outputs)

    if threshold is None:
        threshold = 0.5
    
    # Apply threshold to the probabilities to get binary predictions
    if np.isscalar(threshold):
        predictions = (probabilities > threshold).astype(int)
    else:
        # Ensure threshold is a numpy array and has the same number of elements as the number of classes
        if threshold.size != probabilities.shape[1]:
            raise ValueError("Threshold must have the same number of elements as the number of classes.")

        thresholds_reshaped = threshold.reshape(1, -1)
        # Repeat the threshold for each sample and compare
        predictions = (probabilities > thresholds_reshaped).astype(int)
    return predictions

def convert_labels_to_strings(labels, index_to_tag):
    labels = labels.tolist() if isinstance(labels, torch.Tensor) else labels
    return (index_to_tag[i] for i, label in enumerate(labels) if label == 1)

def convert_labels_to_string(labels, index_to_tag):
    labelstrings = convert_labels_to_strings(labels, index_to_tag)
    return ','.join(labelstrings)

def find_best_threshold(prediction_outputs, true_labels, metric='f1', num_thresholds=100, average='micro'):
    """
    Find the best threshold for binary predictions to optimize the given metric.

    Parameters:
        data_loader (DataLoader): DataLoader for evaluation.
        prediction_outputs (numpy.ndarray): Raw model outputs.
        true_labels (numpy.ndarray): Corresponding true labels.
        metric (str): The metric to optimize ('f1', 'precision', or 'recall').
        num_thresholds (int): The number of threshold values to consider between 0 and 1.
        average (str): The type of averaging performed when computing metrics.

    Returns:
        best_threshold (float): The threshold value that optimizes the given metric.
        best_metric_value (float): The value of the optimized metric at the best threshold.
    """
    best_threshold = None
    best_metric_value = 0

    best_f1, best_precision, best_recall = 0, 0, 0

    for threshold in np.linspace(0, 1, num_thresholds):
        # Get binary predictions based on the current threshold
        predictions_binary = getpredictions_with_threshold(prediction_outputs, threshold)

        # Compute evaluation metrics
        precision, recall, f1 = compute_metrics(true_labels, predictions_binary, average=average)

        # Select the current metric
        if metric == 'f1':
            current_metric_value = f1
        elif metric == 'precision':
            current_metric_value = precision
        elif metric == 'recall':
            current_metric_value = recall
        else:
            raise ValueError("Invalid metric specified. Choose 'f1', 'precision', or 'recall'.")

        # Update the best threshold and metric value if the current one is better
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_threshold = threshold
            best_f1, best_precision, best_recall = f1, precision, recall

    return best_threshold, best_f1, best_precision, best_recall

def find_best_thresholds_per_class(prediction_outputs, true_labels, metric='f1', num_thresholds=100):
    """
    Find the best threshold for binary predictions for each class to optimize the given metric.

    Parameters:
        prediction_outputs (numpy.ndarray): Raw model outputs.
        true_labels (numpy.ndarray): Corresponding true labels.
        metric (str): The metric to optimize ('f1', 'precision', or 'recall').
        num_thresholds (int): The number of threshold values to consider between 0 and 1.

    Returns:
        best_thresholds (numpy.ndarray): The threshold value that optimizes the given metric for each class.
    """
    num_classes = prediction_outputs.shape[1]
    best_thresholds = np.zeros(num_classes)

    # Convert logits to probabilities using the sigmoid function
    probabilities = expit(prediction_outputs)

    for class_idx in range(num_classes):
        best_metric_value_for_class = 0
        best_threshold_for_class = 0.5  # Default threshold

        #logger.debug(f"Optimizing threshold for class {class_idx}.")
        for threshold in np.linspace(0, 1, num_thresholds):
            # Get binary predictions based on the current threshold for this class
            predictions_binary = (probabilities[:, class_idx] > threshold).astype(int)

            # Select the current metric
            if metric == 'f1':
                current_metric_value = sklearnf1(true_labels[:, class_idx], predictions_binary, zero_division=1)
            elif metric == 'precision':
                current_metric_value = precision_score(true_labels[:, class_idx], predictions_binary, zero_division=1)
            elif metric == 'recall':
                current_metric_value = recall_score(true_labels[:, class_idx], predictions_binary, zero_division=1)
            else:
                raise ValueError("Invalid metric specified. Choose 'f1', 'precision', or 'recall'.")
            #logger.debug(f"Class {class_idx}: Threshold {threshold:.4f}, {metric.capitalize()} {current_metric_value:.4f}")

            # Update the best threshold and metric value if the current one is better
            if current_metric_value > best_metric_value_for_class:
                best_metric_value_for_class = current_metric_value
                best_threshold_for_class = threshold

        best_thresholds[class_idx] = best_threshold_for_class
    return best_thresholds

def filter_dict_for_hparams(input_dict):
    """
    Filters out types that arent allowed in hparams for tensorboard
    """
    # Define the allowed types
    allowed_types = (int, float, str, bool, torch.Tensor)
    
    # Create a new dictionary to store the filtered key-value pairs
    filtered_dict = {}
    
    # Iterate over the items in the original dictionary
    for key, value in input_dict.items():
        # If the value is of an allowed type, add it to the new dictionary
        if isinstance(value, allowed_types):
            filtered_dict[key] = value

    return filtered_dict