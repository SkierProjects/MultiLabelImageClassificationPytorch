from sklearn.metrics import f1_score as sklearnf1, precision_score, recall_score, brier_score_loss
import numpy as np
import torch
from imclaslib.logging.loggerfactory import LoggerFactory
from scipy.special import expit
from torch.optim import LBFGS
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

def getConfidences(outputs):
    return torch.sigmoid(outputs).cpu().numpy()

def getpredictions_with_threshold(outputs, device, threshold=0.5):
    """
    Convert raw output scores to binary predictions using a threshold, using numpy arrays.

    Parameters:
    - outputs: numpy.ndarray, raw output scores from the classifier
    - threshold: float, threshold for converting raw scores to binary predictions

    Returns:
    - predictions: numpy.ndarray, binary predictions
    """

    # Apply sigmoid to the outputs to get probabilities
    outputs = torch.Tensor(outputs).to(device)
    probabilities = getConfidences(outputs)

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

def cumulative_uncertainty(probabilities, certainty_window=0.03):
    return np.sum(np.where((probabilities > certainty_window) & (probabilities < (1-certainty_window)),
                                             np.minimum(probabilities - 0.0, 1.0 - probabilities), 0), axis=1)

def uncertainty_metrics(probabilities, certainty_window=0.03):
    # Calculate cumulative uncertainty with the existing function logic
    cumulative_uncertainties = cumulative_uncertainty(probabilities, certainty_window)

    # Initialize lists to store the additional metrics
    max_uncertainties = []
    mean_uncertainties = []
    mean_entropies = []

    # Calculate additional metrics for each image
    for image_probs in probabilities:
        # Calculate the uncertainty for each class probability in the image
        uncertainties = np.where(
            (image_probs > certainty_window) & (image_probs < (1 - certainty_window)),
            np.minimum(image_probs, 1.0 - image_probs),
            0
        )

        # Calculate the entropy for each class probability in the image
        eps = np.finfo(float).eps  # Avoid division by zero in log
        image_entropies = -(
            image_probs * np.log(np.clip(image_probs, eps, 1)) +
            (1 - image_probs) * np.log(np.clip(1 - image_probs, eps, 1))
        ) / np.log(2)  # Normalize to log base 2

        # Calculate the mean entropy for the image
        mean_entropy = np.mean(image_entropies)

        # Find the max uncertainty (which is the max distance from certainty) for the image
        max_uncertainty_index = np.argmax(uncertainties)
        max_uncertainty = uncertainties[max_uncertainty_index]

        # Store the max uncertainty and its corresponding class number
        max_uncertainties.append((max_uncertainty, max_uncertainty_index))

        # Calculate the mean uncertainty for the image
        mean_uncertainty = np.mean(uncertainties)

        # Append the mean entropy and mean uncertainty to their respective lists
        mean_entropies.append(mean_entropy)
        mean_uncertainties.append(mean_uncertainty)

    # Return the results as a dictionary
    return {
        'cumulative_uncertainties': cumulative_uncertainties,
        'max_uncertainties': max_uncertainties,
        'mean_uncertainties': mean_uncertainties,
        'mean_entropies': mean_entropies
    }

def find_best_threshold(prediction_outputs, true_labels, device, metric='f1', num_thresholds=100, average='micro'):
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
        predictions_binary = getpredictions_with_threshold(prediction_outputs, device, threshold)

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

def multi_label_brier_score(y_true, y_pred, average='macro'):
    """
    Calculate the Brier score for multi-label classification with various averaging methods.

    :param y_true: A NumPy array of ground truth labels with shape (num_samples, num_classes).
    :param y_pred: A NumPy array of predicted logits with shape (num_samples, num_classes).
    :param average: The averaging method to use ('macro', 'micro', 'weighted', 'samples').
    :return: The Brier score.
    """
    y_pred = getConfidences(y_pred)
    num_classes = y_true.shape[1]
    
    if average == 'macro':
        # Calculate Brier score for each label and then average
        brier_scores = [np.mean((y_true[:, i] - y_pred[:, i]) ** 2) for i in range(y_true.shape[1])]
        return np.mean(brier_scores)
    
    elif average == 'micro':
        # Calculate the mean squared difference between all true labels and predictions
        return np.mean((y_true - y_pred) ** 2)
    
    elif average == 'weighted':
        # Calculate Brier score for each label, weighted by support (the number of true instances for each label)
        supports = np.sum(y_true, axis=0)
        brier_scores = [np.mean((y_true[:, i] - y_pred[:, i]) ** 2) for i in range(y_true.shape[1])]
        return np.average(brier_scores, weights=supports)
    
    elif average == 'samples':
        # Calculate Brier score for each individual label within each sample and average these
        brier_scores = np.mean((y_true - y_pred) ** 2, axis=1)
        return np.mean(brier_scores)
    
    else:
        raise ValueError("The 'average' parameter should be one of 'macro', 'micro', 'weighted', or 'samples'.")
    
def temperature_scale(logits, temperature):
    """
    Scale the logits by the temperature.
    """
    return logits / temperature

def find_optimal_temperature(valid_logits, valid_labels, device):
    """
    Find the optimal temperature for multilabel classification using the validation set.
    """
    # Initial temperature
    temperature = torch.nn.Parameter(torch.ones(1, device=device))
    
    # Define the loss function and optimizer
    bce_with_logits_loss = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval():
        loss = bce_with_logits_loss(temperature_scale(valid_logits, temperature), valid_labels)
        loss.backward()
        return loss
    
    # Find the optimal temperature
    optimizer.step(eval)
    
    return temperature.item()