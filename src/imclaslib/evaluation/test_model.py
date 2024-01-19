from imclaslib.evaluation.modelevaluator import ModelEvaluator
from imclaslib.logging.loggerfactory import LoggerFactory
from imclaslib.tensorboard.tensorboardwriter import TensorBoardWriter
from imclaslib.metrics import metricutils
import torch
import imclaslib.dataset.datasetutils as datasetutils
import time
import numpy as np
from scipy.special import expit

# Set up logging for the training process
logger = LoggerFactory.get_logger(f"logger.{__name__}")

def evaluate_model(this_config):
    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = datasetutils.get_data_loader_by_name("test", config=this_config)
    valid_loader = datasetutils.get_data_loader_by_name("valid", config=this_config)
    valid_test_loader = datasetutils.get_data_loader_by_name("valid+test", config=this_config, shuffle=True)

    # intialize the model
    with get_model_evaluator(this_config, device) as modelEvaluator:
        epochs = modelEvaluator.model_data["epoch"]

        valid_start_time = time.time()
        valid_results = modelEvaluator.predict(valid_loader)
        valid_end_time = time.time()

        test_start_time = time.time()
        test_results = modelEvaluator.predict(test_loader)
        test_end_time = time.time()

        valid_test_start_time = time.time()       
        validtest_results = modelEvaluator.predict(valid_test_loader)
        valid_test_end_time = time.time()


        valid_predictions, valid_correct_labels, valid_loss = valid_results['predictions'], valid_results['true_labels'], valid_results['avg_loss']
        test_predictions, test_correct_labels, test_loss = test_results['predictions'], test_results['true_labels'], test_results['avg_loss']
        validtest_predictions, validtest_correct_labels, validtest_loss = validtest_results['predictions'], validtest_results['true_labels'], validtest_results['avg_loss']
        confidence_thresholds = (0.01, 0.2, 0.4, 0.7, 1.0)  # Adjust these thresholds to suit your needs
        test_proabilities = expit(test_predictions)
        test_confidence_categories = categorize_predictions(test_proabilities, confidence_thresholds)

        valid_elapsed_time = valid_end_time - valid_start_time
        test_elapsed_time = test_end_time - test_start_time
        valid_test_elapsed_time = valid_test_end_time - valid_test_start_time

        valid_num_images = len(valid_loader.dataset)
        test_num_images = len(test_loader.dataset)
        valid_test_num_images = len(valid_test_loader.dataset)

        valid_images_per_second = valid_num_images / valid_elapsed_time
        test_images_per_second = test_num_images / test_elapsed_time
        valid_test_images_per_second = valid_test_num_images / valid_test_elapsed_time

        avg_images_per_second = (valid_num_images + test_num_images + valid_test_num_images) / (valid_elapsed_time + test_elapsed_time + valid_test_elapsed_time)

        logger.info(f"Validation Img/sec: {valid_images_per_second}")
        logger.info(f"Test Img/sec: {test_images_per_second}")
        logger.info(f"Validation+Test Img/sec: {valid_test_images_per_second}")
        logger.info(f"Avg Img/sec: {avg_images_per_second}")

        logger.info(f"Validation Loss: {valid_loss}")
        logger.info(f"Test Loss: {test_loss}")
        logger.info(f"Validation+Test Loss: {validtest_loss}")

        val_f1_default, val_precision_default, val_recall_default = modelEvaluator.evaluate_predictions(valid_loader, valid_predictions, valid_correct_labels, epochs, threshold=0.5, average="micro")
        test_f1_default, test_precision_default, test_recall_default = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=0.5, average="micro")
        validtest_f1_default, validtest_precision_default, validtest_recall_default = modelEvaluator.evaluate_predictions(valid_test_loader, validtest_predictions, validtest_correct_labels, epochs, threshold=0.5, average="micro")

        logger.info(f"Validation Default F1: F1: {val_f1_default}, Precision: {val_precision_default}, Recall: {val_recall_default} at Threshold: 0.5")
        logger.info(f"Test Default F1: F1: {test_f1_default}, Precision: {test_precision_default}, Recall: {test_recall_default} at Threshold: 0.5")
        logger.info(f"Valid+Test Default F1: F1: {validtest_f1_default}, Precision: {validtest_precision_default}, Recall: {validtest_recall_default} at Threshold: 0.5")


        val_best_f1_threshold, val_f1_valoptimized, val_precision_valoptimized, val_recall_valoptimized = metricutils.find_best_threshold(valid_predictions, valid_correct_labels, "f1")
        logger.info(f"Validation Best F1: F1: {val_f1_valoptimized}, Precision: {val_precision_valoptimized}, Recall: {val_recall_valoptimized} at Threshold:{val_best_f1_threshold}")
        test_f1_valoptimized, test_precision_valoptimized, test_recall_valoptimized = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=val_best_f1_threshold, average="micro", datasetSubset="Test", metricMode="Test")
        validtest_f1_valoptimized, validtest_precision_valoptimized, validtest_recall_valoptimized = modelEvaluator.evaluate_predictions(valid_test_loader, validtest_predictions, validtest_correct_labels, epochs, threshold=val_best_f1_threshold, average="micro")
        logger.info(f"Test Best F1 (measured from Val): F1: {test_f1_valoptimized}, Precision: {test_precision_valoptimized}, Recall: {test_recall_valoptimized} at Threshold:{val_best_f1_threshold}")
        logger.info(f"Valid+Test Best F1 (measured from Val): F1: {validtest_f1_valoptimized}, Precision: {validtest_precision_valoptimized}, Recall: {validtest_recall_valoptimized} at Threshold:{val_best_f1_threshold}")

        best_f1_thresholds_per_class = metricutils.find_best_thresholds_per_class(valid_predictions, valid_correct_labels)
        test_f1_valoptimizedperclass, test_precision_valoptimizedperclass, test_recall_valoptimizedperclass = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=best_f1_thresholds_per_class, average="micro")
        logger.info(f"Test Best F1 Per Class (Val Optimized): F1: {test_f1_valoptimizedperclass}, Precision: {test_precision_valoptimizedperclass}, Recall: {test_recall_valoptimizedperclass} at Threshold:{best_f1_thresholds_per_class}")

        hparams = metricutils.filter_dict_for_hparams(modelEvaluator.model_data)
        final_metrics = {
            'F1/Default/Validation': val_f1_default,
            'F1/Default/Test': test_f1_default,
            'F1/Default/Valid+Test': validtest_f1_default,
            'F1/ValOptimizedThreshold/Validation': val_f1_valoptimized,
            'F1/ValOptimizedThreshold/Test': test_f1_valoptimized,
            'F1/ValOptimizedThreshold/Valid+Test': validtest_f1_valoptimized,
            'Precision/Default/Validation': val_precision_default,
            'Precision/Default/Test': test_precision_default,
            'Precision/Default/Valid+Test': validtest_precision_default,
            'Precision/ValOptimizedThreshold/Validation': val_precision_valoptimized,
            'Precision/ValOptimizedThreshold/Test': test_precision_valoptimized,
            'Precision/ValOptimizedThreshold/Valid+Test': validtest_precision_valoptimized,
            'Recall/Default/Validation': val_recall_default,
            'Recall/Default/Test': test_recall_default,
            'Recall/Default/Valid+Test': validtest_recall_default,
            'Recall/ValOptimizedThreshold/Validation': val_recall_valoptimized,
            'Recall/ValOptimizedThreshold/Test': test_recall_valoptimized,
            'Recall/ValOptimizedThreshold/Valid+Test': validtest_recall_valoptimized,
            'F1/ValOptimizedThresholdPerClass/Test': test_f1_valoptimizedperclass,
            'Precision/ValOptimizedThresholdPerClass/Test': test_precision_valoptimizedperclass,
            'Recall/ValOptimizedThresholdPerClass/Test': test_recall_valoptimizedperclass,
            'ImagesPerSecond/Validation': valid_images_per_second,
            'ImagesPerSecond/Test': test_images_per_second,
            'ImagesPerSecond/Valid+Test': valid_test_images_per_second,
            'ImagesPerSecond/Average': avg_images_per_second,
            'Loss/TrainOverTest+ValidRatio': modelEvaluator.model_data["train_loss"] / validtest_loss
        }
        modelEvaluator.tensorBoardWriter.add_scalars_from_dict(final_metrics, epochs)
        modelEvaluator.tensorBoardWriter.add_hparams(hparams, final_metrics)

        test_f1s_per_class, _, _ =  modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=val_best_f1_threshold, average=None)
        tagmappings = datasetutils.get_index_to_tag_mapping(this_config)
        for class_index in range(this_config.num_classes):
            modelEvaluator.tensorBoardWriter.add_scalar(f'F1_Class_{tagmappings[class_index]}/ValOptimizedThreshold/Test', test_f1s_per_class[class_index], epochs)

        val_test_f1s_per_class, _, _ =  modelEvaluator.evaluate_predictions(valid_test_loader, validtest_predictions, validtest_correct_labels, epochs, threshold=0.5, average=None)
        tagmappings = datasetutils.get_index_to_tag_mapping(this_config)
        for class_index in range(this_config.num_classes):
            modelEvaluator.tensorBoardWriter.add_scalar(f'F1_Class_{tagmappings[class_index]}/ValOptimizedThreshold/Valid+Test', val_test_f1s_per_class[class_index], epochs)

        # Prepare to store results by category
        f1_scores_by_category = []
        samples_by_category = []
        # Calculate F1 scores for each category
        for category in range(len(confidence_thresholds)+1):
            category_mask = (test_confidence_categories == category)
            category_predictions = test_predictions[category_mask]
            category_true_labels = test_correct_labels[category_mask]
            
            # Ensure there are samples in the category before calculating F1
            if category_predictions.shape[0] > 0:
                binary_predictions = (category_predictions > val_best_f1_threshold).astype(int)
                category_f1 = metricutils.f1_score(category_true_labels, binary_predictions)

                binary_predictions_default = (category_predictions > 0.5).astype(int)
            else:
                category_f1 = None

            f1_scores_by_category.append(category_f1)
            samples_by_category.append(np.sum(category_mask))

            # Log results
            logger.info(f"Confidence Category {category} - Images: {samples_by_category[-1]}, F1 Score: {f1_scores_by_category[-1]}")

            # Log to TensorBoard
            modelEvaluator.tensorBoardWriter.add_scalar(f'F1_Score_By_Confidence_Category/Category_{category}', f1_scores_by_category[-1] if category_f1 else 0, epochs)
            modelEvaluator.tensorBoardWriter.add_scalar(f'Samples_By_Confidence_Category/Category_{category}', samples_by_category[-1] if category_f1 else 0, epochs)
        assert np.sum(samples_by_category) == test_num_images

def get_model_evaluator(config, device):
    if config.ensemble_model_configs:
        return ModelEvaluator.from_ensemble(device, config, TensorBoardWriter(config=config))
    else:
        return ModelEvaluator.from_file(device, config, TensorBoardWriter(config=config))
    
def categorize_predictions(probabilities, thresholds, certainty_window=0.03):
    """
    Categorize images into confidence levels based on the distance from the decision boundary for each label.

    Parameters:
    - probabilities: numpy.ndarray, the predicted probabilities for each label of each image
    - thresholds: tuple, containing the confidence thresholds for categorization

    Returns:
    - categories: numpy.ndarray, the categories for each image
    """

    cumulative_uncertainty = np.sum(np.where((probabilities > certainty_window) & (probabilities < (1-certainty_window)),
                                             np.minimum(probabilities - 0.0, 1.0 - probabilities), 0), axis=1)
    
    # Calculate and print the mean and standard deviation of cumulative uncertainties for debugging
    mean_uncertainty = np.mean(cumulative_uncertainty)
    std_uncertainty = np.std(cumulative_uncertainty)
    print("Mean cumulative uncertainty:", mean_uncertainty)
    print("Standard deviation of cumulative uncertainty:", std_uncertainty)
    
    # Assign confidence categories based on image confidence levels
    categories = np.digitize(cumulative_uncertainty, thresholds)
    
    return categories