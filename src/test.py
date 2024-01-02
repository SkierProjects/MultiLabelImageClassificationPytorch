from config import config
import utils.files.pathutils as pathutils
# Set up system path for relative imports
pathutils.setup_sys_path()
from utils.evaluation.modelevaluator import ModelEvaluator
from utils.logging.loggerfactory import LoggerFactory
from utils.tensorboard.tensorboardwriter import TensorBoardWriter
from utils.metrics import metricutils
import torch
import utils.dataset.datasetutils as datasetutils
# Set up logging for the training process
logger = LoggerFactory.setup_logging("logger", log_file=pathutils.combine_path(
    pathutils.get_log_dir_path(), 
    f"{config.model_name}_{config.image_size}_{config.model_weights}",
    f"train__{pathutils.get_datetime()}.log"))

def evaluate_model(this_config=config):
    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = datasetutils.get_data_loader_by_name("test", config=this_config)
    valid_loader = datasetutils.get_data_loader_by_name("valid", config=this_config)
    valid_test_loader = datasetutils.get_data_loader_by_name("valid+test", config=this_config, shuffle=True)

    # intialize the model
    with ModelEvaluator.from_file(device, TensorBoardWriter(config=this_config), this_config) as modelEvaluator:
        epochs = modelEvaluator.epochs
        valid_results = modelEvaluator.predict(valid_loader)
        test_results = modelEvaluator.predict(test_loader)
        validtest_results = modelEvaluator.predict(valid_test_loader)

        valid_predictions, valid_correct_labels, valid_loss = valid_results['predictions'], valid_results['true_labels'], valid_results['avg_loss']
        test_predictions, test_correct_labels, test_loss = test_results['predictions'], test_results['true_labels'], test_results['avg_loss']
        validtest_predictions, validtest_correct_labels, validtest_loss = validtest_results['predictions'], validtest_results['true_labels'], validtest_results['avg_loss']

        logger.info(f"Validation Loss: {valid_loss}")
        logger.info(f"Test Loss: {test_loss}")
        logger.info(f"Validation+Test Loss: {validtest_loss}")

        val_f1_default, val_precision_default, val_recall_default = modelEvaluator.evaluate_predictions(valid_loader, valid_predictions, valid_correct_labels, epochs, "Valid", threshold=0.5, average="micro")
        test_f1_default, test_precision_default, test_recall_default = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, "Test", threshold=0.5, average="micro")
        validtest_f1_default, validtest_precision_default, validtest_recall_default = modelEvaluator.evaluate_predictions(valid_test_loader, validtest_predictions, validtest_correct_labels, epochs, "Valid+Test", threshold=0.5, average="micro")

        logger.info(f"Validation Default F1: F1: {val_f1_default}, Precision: {val_precision_default}, Recall: {val_recall_default} at Threshold: 0.5")
        logger.info(f"Test Default F1: F1: {test_f1_default}, Precision: {test_precision_default}, Recall: {test_recall_default} at Threshold: 0.5")
        logger.info(f"Valid+Test Default F1: F1: {validtest_f1_default}, Precision: {validtest_precision_default}, Recall: {validtest_recall_default} at Threshold: 0.5")


        val_best_f1_threshold, val_f1, val_precision, val_recall = metricutils.find_best_threshold(valid_predictions, valid_correct_labels, "f1")
        logger.info(f"Validation Best F1: F1: {val_f1}, Precision: {val_precision}, Recall: {val_recall} at Threshold:{val_best_f1_threshold}")
        test_f1, test_precision, test_recall = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, "Valid", threshold=val_best_f1_threshold, average="micro")
        validtest_f1, validtest_precision, validtest_recall = modelEvaluator.evaluate_predictions(valid_test_loader, validtest_predictions, validtest_correct_labels, epochs, "Valid", threshold=val_best_f1_threshold, average="micro")
        logger.info(f"Test Best F1 (measured from Val): F1: {test_f1}, Precision: {test_precision}, Recall: {test_recall} at Threshold:{val_best_f1_threshold}")
        logger.info(f"Valid+Test Best F1 (measured from Val): F1: {validtest_f1}, Precision: {validtest_precision}, Recall: {validtest_recall} at Threshold:{val_best_f1_threshold}")

        best_f1_thresholds_per_class = metricutils.find_best_thresholds_per_class(valid_predictions, valid_correct_labels)
        val_f1, val_precision, val_recall = modelEvaluator.evaluate_predictions(valid_loader, valid_predictions, valid_correct_labels, epochs, "Valid", threshold=best_f1_thresholds_per_class, average="micro")
        logger.info(f"Validation Best F1 Per Class: F1: {val_f1}, Precision: {val_precision}, Recall: {val_recall} at Threshold:{best_f1_thresholds_per_class}")
        test_f1, test_precision, test_recall = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, epochs, "Valid", threshold=best_f1_thresholds_per_class, average="micro")
        validtest_f1, validtest_precision, validtest_recall = modelEvaluator.evaluate_predictions(valid_test_loader, validtest_predictions, validtest_correct_labels, epochs, "Valid", threshold=best_f1_thresholds_per_class, average="micro")
        logger.info(f"Test Best F1 Per Class (measured from Val): F1: {test_f1}, Precision: {test_precision}, Recall: {test_recall} at Threshold:{val_best_f1_threshold}")
        logger.info(f"Valid+Test Best F1 Per Class (measured from Val): F1: {validtest_f1}, Precision: {validtest_precision}, Recall: {validtest_recall} at Threshold:{val_best_f1_threshold}")



def main():
    # Call the train_model function with the configuration object
    evaluate_model(config)

if __name__ == '__main__':
    main()