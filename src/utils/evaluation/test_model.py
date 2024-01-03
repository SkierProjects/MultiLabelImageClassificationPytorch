from config import config
from utils.evaluation.modelevaluator import ModelEvaluator
from utils.logging.loggerfactory import LoggerFactory
from utils.tensorboard.tensorboardwriter import TensorBoardWriter
from utils.metrics import metricutils
import torch
import utils.dataset.datasetutils as datasetutils
import time
# Set up logging for the training process
logger = LoggerFactory.get_logger(f"logger.{__name__}")

def evaluate_model(this_config=config):
    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = datasetutils.get_data_loader_by_name("test", config=this_config)
    valid_loader = datasetutils.get_data_loader_by_name("valid", config=this_config)
    valid_test_loader = datasetutils.get_data_loader_by_name("valid+test", config=this_config, shuffle=True)

    # intialize the model
    with ModelEvaluator.from_file(device, this_config, TensorBoardWriter(config=this_config)) as modelEvaluator:
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

        valid_elapsed_time = valid_end_time - valid_start_time
        test_elapsed_time = test_end_time - test_start_time
        valid_test_elapsed_time = valid_test_end_time - valid_test_start_time

        valid_num_images = len(valid_loader.dataset)
        test_num_images = len(test_loader.dataset)
        valid_test_num_images = len(valid_test_loader.dataset)

        valid_images_per_second = valid_num_images / valid_elapsed_time
        test_images_per_second = test_num_images / test_elapsed_time
        valid_test_images_per_second = valid_test_num_images / valid_test_elapsed_time

        avg_images_per_second = (valid_images_per_second + test_images_per_second + valid_test_images_per_second) / 3

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
            'ImagesPerSecond/Average': avg_images_per_second
        }
        modelEvaluator.tensorBoardWriter.add_scalars_from_dict(final_metrics, epochs)
        modelEvaluator.tensorBoardWriter.add_hparams(hparams, final_metrics)

        test_f1s_per_class, _, _ =  modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, threshold=val_best_f1_threshold, average=None)
        tagmappings = datasetutils.get_dataset_tag_mappings(this_config)
        for class_index in range(this_config.num_classes):
            modelEvaluator.tensorBoardWriter.add_scalar(f'F1_Class_{tagmappings[class_index]}/ValOptimizedThreshold/Test', test_f1s_per_class[class_index], epochs)