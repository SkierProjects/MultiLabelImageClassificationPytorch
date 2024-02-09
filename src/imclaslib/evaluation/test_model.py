import wandb
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

    # intialize the model
    with get_model_evaluator(this_config, device) as modelEvaluator:
        epochs = modelEvaluator.model_data["epoch"]
        #modelEvaluator.compile()
        valid_start_time = time.time()
        valid_results = modelEvaluator.predict(valid_loader)
        valid_end_time = time.time()

        test_start_time = time.time()
        test_results = modelEvaluator.predict(test_loader)
        test_end_time = time.time()

        valid_predictions, valid_correct_labels, valid_loss = valid_results['predictions'], valid_results['true_labels'], valid_results['avg_loss']
        test_predictions, test_correct_labels, test_loss = test_results['predictions'], test_results['true_labels'], test_results['avg_loss']

        # Assuming device is already defined as in your code snippet
        # Perform temperature scaling on validation logits
        valid_logits = torch.Tensor(valid_predictions).to(device)
        valid_labels = torch.Tensor(valid_correct_labels).to(device)  # Assuming valid_correct_labels are floats in [0, 1]
        optimal_temperature = metricutils.find_optimal_temperature(valid_logits, valid_labels, device)
        
        valid_predictions = metricutils.temperature_scale(valid_logits, optimal_temperature)
        # Apply temperature scaling to test logits
        test_logits = torch.Tensor(test_predictions).to(device)
        test_predictions = metricutils.temperature_scale(test_logits, optimal_temperature)

        confidence_thresholds = (0.01, 0.2, 0.4, 0.7, 1.0)  # Adjust these thresholds to suit your needs
        test_proabilities = metricutils.getConfidences(test_predictions) #torch.sigmoid(test_predictions).cpu().numpy()
        test_confidence_categories = categorize_predictions(test_proabilities, confidence_thresholds)

        valid_elapsed_time = valid_end_time - valid_start_time
        test_elapsed_time = test_end_time - test_start_time

        valid_num_images = len(valid_loader.dataset)
        test_num_images = len(test_loader.dataset)

        valid_images_per_second = valid_num_images / valid_elapsed_time
        test_images_per_second = test_num_images / test_elapsed_time

        avg_images_per_second = (valid_num_images + test_num_images) / (valid_elapsed_time + test_elapsed_time)

        logger.info(f"Validation Img/sec: {valid_images_per_second}")
        logger.info(f"Test Img/sec: {test_images_per_second}")

        logger.info(f"Avg Img/sec: {avg_images_per_second}")

        logger.info(f"Validation Loss: {valid_loss}")
        logger.info(f"Test Loss: {test_loss}")

        val_f1_default_micro, val_precision_default_micro, val_recall_default_micro = modelEvaluator.evaluate_predictions(valid_loader, valid_predictions, valid_correct_labels, epochs, threshold=0.5, average="micro")
        test_f1_default_micro, test_precision_default_micro, test_recall_default_micro = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=0.5, average="micro")
        
        val_f1_default_macro, val_precision_default_macro, val_recall_default_macro = modelEvaluator.evaluate_predictions(valid_loader, valid_predictions, valid_correct_labels, epochs, threshold=0.5, average="macro")
        test_f1_default_macro, test_precision_default_macro, test_recall_default_macro = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=0.5, average="macro")

        val_f1_default_weighted, val_precision_default_weighted, val_recall_default_weighted = modelEvaluator.evaluate_predictions(valid_loader, valid_predictions, valid_correct_labels, epochs, threshold=0.5, average="weighted")
        test_f1_default_weighted, test_precision_default_weighted, test_recall_default_weighted = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=0.5, average="weighted")

        val_f1_default_samples, val_precision_default_samples, val_recall_default_samples = modelEvaluator.evaluate_predictions(valid_loader, valid_predictions, valid_correct_labels, epochs, threshold=0.5, average="samples")
        test_f1_default_samples, test_precision_default_samples, test_recall_default_samples = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=0.5, average="samples")

        valid_brier_micro = metricutils.multi_label_brier_score(valid_correct_labels, valid_predictions, 'micro')
        valid_brier_macro = metricutils.multi_label_brier_score(valid_correct_labels, valid_predictions, 'macro')
        valid_brier_weighted = metricutils.multi_label_brier_score(valid_correct_labels, valid_predictions, 'weighted')
        valid_brier_samples = metricutils.multi_label_brier_score(valid_correct_labels, valid_predictions, 'samples')
        logger.info(f"Brier Score for Validation: Macro:{valid_brier_macro}, Micro:{valid_brier_micro}, Weighted:{valid_brier_weighted}, Samples:{valid_brier_samples}")

        test_brier_micro = metricutils.multi_label_brier_score(test_correct_labels, test_predictions, 'micro')
        test_brier_macro = metricutils.multi_label_brier_score(test_correct_labels, test_predictions, 'macro')
        test_brier_weighted = metricutils.multi_label_brier_score(test_correct_labels, test_predictions, 'weighted')
        test_brier_samples = metricutils.multi_label_brier_score(test_correct_labels, test_predictions, 'samples')
        logger.info(f"Brier Score for Test: Macro:{test_brier_macro}, Micro:{test_brier_micro}, Weighted:{test_brier_weighted}, Samples:{test_brier_samples}")

        logger.info(f"Validation Default F1: F1: {val_f1_default_micro}, Precision: {val_precision_default_micro}, Recall: {val_recall_default_micro} at Threshold: 0.5")
        logger.info(f"Test Default F1: F1: {test_f1_default_micro}, Precision: {test_precision_default_micro}, Recall: {test_recall_default_micro} at Threshold: 0.5")

        val_best_f1_threshold, val_f1_valoptimized, val_precision_valoptimized, val_recall_valoptimized = metricutils.find_best_threshold(valid_predictions, valid_correct_labels, device, "f1")
        logger.info(f"Validation Best F1: F1: {val_f1_valoptimized}, Precision: {val_precision_valoptimized}, Recall: {val_recall_valoptimized} at Threshold:{val_best_f1_threshold}")
        test_f1_valoptimized, test_precision_valoptimized, test_recall_valoptimized = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=val_best_f1_threshold, average="micro", datasetSubset="Test", metricMode="Test")

        test_f1_valoptimized_macro, _, _ = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=val_best_f1_threshold, average="macro", datasetSubset="Test", metricMode="Test")
        test_f1_valoptimized_weighted, _, _ = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=val_best_f1_threshold, average="weighted", datasetSubset="Test", metricMode="Test")
        test_f1_valoptimized_samples, _, _ = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=val_best_f1_threshold, average="samples", datasetSubset="Test", metricMode="Test")
        logger.info(f"Test Best F1 (measured from Val): F1: {test_f1_valoptimized}, Precision: {test_precision_valoptimized}, Recall: {test_recall_valoptimized} at Threshold:{val_best_f1_threshold}")
        logger.info(f"Test Best F1 (measured from Val): F1 Macro: {test_f1_valoptimized_macro}, F1 Weighted: {test_f1_valoptimized_weighted}, F1 Samples: {test_f1_valoptimized_samples} at Threshold:{val_best_f1_threshold}")

        #best_f1_thresholds_per_class = metricutils.find_best_thresholds_per_class(valid_predictions, valid_correct_labels)
        #test_f1_valoptimizedperclass, test_precision_valoptimizedperclass, test_recall_valoptimizedperclass = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=best_f1_thresholds_per_class, average="micro")
        #logger.info(f"Test Best F1 Per Class (Val Optimized): F1: {test_f1_valoptimizedperclass}, Precision: {test_precision_valoptimizedperclass}, Recall: {test_recall_valoptimizedperclass} at Threshold:{best_f1_thresholds_per_class}")

        hparams = metricutils.filter_dict_for_hparams(modelEvaluator.model_data)
        final_metrics = {
            'F1/Default/Validation/Micro': val_f1_default_micro,
            'F1/Default/Validation/Macro': val_f1_default_macro,
            'F1/Default/Validation/Weighted': val_f1_default_weighted,
            'F1/Default/Validation/Samples': val_f1_default_samples,
            'F1/Default/Test/Micro': test_f1_default_micro,
            'F1/Default/Test/Macro': test_f1_default_macro,
            'F1/Default/Test/Weighted': test_f1_default_weighted,
            'F1/Default/Test/Samples': test_f1_default_samples,
            'F1/ValOptimizedThreshold/Validation': val_f1_valoptimized,
            'F1/ValOptimizedThreshold/Test/Micro': test_f1_valoptimized,
            'F1/ValOptimizedThreshold/Test/Macro': test_f1_valoptimized_macro,
            'F1/ValOptimizedThreshold/Test/Weighted': test_f1_valoptimized_weighted,
            'F1/ValOptimizedThreshold/Test/Samples': test_f1_valoptimized_samples,
            'Precision/Default/Validation': val_precision_default_micro,
            'Precision/Default/Test': test_precision_default_micro,
            'Precision/ValOptimizedThreshold/Validation': val_precision_valoptimized,
            'Precision/ValOptimizedThreshold/Test': test_precision_valoptimized,
            'Recall/Default/Validation': val_recall_default_micro,
            'Recall/Default/Test': test_recall_default_micro,
            'Recall/ValOptimizedThreshold/Validation': val_recall_valoptimized,
            'Recall/ValOptimizedThreshold/Test': test_recall_valoptimized,
            'ImagesPerSecond/Validation': valid_images_per_second,
            'ImagesPerSecond/Test': test_images_per_second,
            'ImagesPerSecond/Average': avg_images_per_second,
            'Loss/TrainOverTestRatio': modelEvaluator.model_data["train_loss"] / test_loss,
            'Brier/Validation/Micro': valid_brier_micro,
            'Brier/Validation/Macro': valid_brier_macro,
            'Brier/Validation/Weighted': valid_brier_weighted,
            'Brier/Validation/Samples': valid_brier_samples,
            'Brier/Test/Micro': test_brier_micro,
            'Brier/Test/Macro': test_brier_macro,
            'Brier/Test/Weighted': test_brier_weighted,
            'Brier/Test/Samples': test_brier_samples,
            'Temperature/ValOptimized': optimal_temperature
        }
        wandb.log(final_metrics)
        modelEvaluator.tensorBoardWriter.add_scalars_from_dict(final_metrics, epochs)
        modelEvaluator.tensorBoardWriter.add_hparams(hparams, final_metrics)

        test_f1s_per_class, test_precision_per_class, test_recall_per_class =  modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=val_best_f1_threshold, average=None)
        tagmappings = datasetutils.get_index_to_tag_mapping(this_config)

        #val_test_f1s_per_class, _, _ =  modelEvaluator.evaluate_predictions(valid_test_loader, validtest_predictions, validtest_correct_labels, epochs, threshold=0.5, average=None)
        tagmappings = datasetutils.get_index_to_tag_mapping(this_config)
        testF1s = []

        annotationCounts, fileCounts = datasetutils.analyze_csv(this_config)
        for class_index in range(this_config.model_num_classes):
            #modelEvaluator.tensorBoardWriter.add_scalar(f'F1_Class_{tagmappings[class_index]}/ValOptimizedThreshold/Valid+Test', val_test_f1s_per_class[class_index], epochs)
            modelEvaluator.tensorBoardWriter.add_scalar(f'F1_Class_{tagmappings[class_index]}/ValOptimizedThreshold/Test', test_f1s_per_class[class_index], epochs)
            testF1s.append([tagmappings[class_index], test_f1s_per_class[class_index], test_precision_per_class[class_index], test_recall_per_class[class_index], annotationCounts[tagmappings[class_index]]])
        my_table = wandb.Table(columns=["ClassName", "ClassF1", "ClassPrecision", "ClassRecall", "ClassDatasetCount"], data=testF1s)
        wandb.log({"F1_Scores_by_Class": my_table})

        wandb.log({"Dataset/Stats": fileCounts})
        # Prepare to store results by category
        f1_scores_by_category = []
        samples_by_category = []

        data = []
        confidence_thresholds_len = len(confidence_thresholds)
        # Calculate F1 scores for each category
        for category in range(confidence_thresholds_len+1):
            category_mask = (test_confidence_categories == category)
            category_predictions = test_proabilities[category_mask]
            category_true_labels = test_correct_labels[category_mask]
            
            # Ensure there are samples in the category before calculating F1
            if category_predictions.shape[0] > 0:
                binary_predictions = (category_predictions > val_best_f1_threshold).astype(int)
                category_f1 = metricutils.f1_score(category_true_labels, binary_predictions)
            else:
                category_f1 = None

            f1_scores_by_category.append(category_f1)
            samples_by_category.append(np.sum(category_mask))

            # Log results
            logger.info(f"Confidence Category {category} - Images: {samples_by_category[-1]}, F1 Score: {f1_scores_by_category[-1]}")

            # Log to TensorBoard
            modelEvaluator.tensorBoardWriter.add_scalar(f'F1_Score_By_Confidence_Category/Category_{category}', f1_scores_by_category[-1] if category_f1 else 0, epochs)
            modelEvaluator.tensorBoardWriter.add_scalar(f'Samples_By_Confidence_Category/Category_{category}', samples_by_category[-1] if category_f1 else 0, epochs)
            data.append([confidence_thresholds[category] if category < confidence_thresholds_len else 1000, f1_scores_by_category[-1] if category_f1 else 0, samples_by_category[-1] if category_f1 else 0])
        assert np.sum(samples_by_category) == test_num_images
        my_table2 = wandb.Table(columns=["Confidence Threshold", "F1 Score", "Sample Count"], data=data)
        wandb.log({"Data by Categories of Confidence": my_table2})

def get_model_evaluator(config, device):
    if config.model_ensemble_model_configs:
        return ModelEvaluator.from_ensemble(device, config, TensorBoardWriter(config=config))
    else:
        return ModelEvaluator.from_file(device, config, TensorBoardWriter(config=config))
    
def cumulative_uncertainty(probabilities, certainty_window=0.03):
    return np.sum(np.where((probabilities > certainty_window) & (probabilities < (1-certainty_window)),
                                             np.minimum(probabilities - 0.0, 1.0 - probabilities), 0), axis=1)

def categorize_predictions(probabilities, thresholds, certainty_window=0.03):
    """
    Categorize images into confidence levels based on the distance from the decision boundary for each label.

    Parameters:
    - probabilities: numpy.ndarray, the predicted probabilities for each label of each image
    - thresholds: tuple, containing the confidence thresholds for categorization

    Returns:
    - categories: numpy.ndarray, the categories for each image
    """

    cumulative_uncertainty = metricutils.cumulative_uncertainty(probabilities, certainty_window)
    
    # Calculate and print the mean and standard deviation of cumulative uncertainties for debugging
    mean_uncertainty = np.mean(cumulative_uncertainty)
    std_uncertainty = np.std(cumulative_uncertainty)
    print("Mean cumulative uncertainty:", mean_uncertainty)
    print("Standard deviation of cumulative uncertainty:", std_uncertainty)
    
    # Assign confidence categories based on image confidence levels
    categories = np.digitize(cumulative_uncertainty, thresholds)
    
    return categories