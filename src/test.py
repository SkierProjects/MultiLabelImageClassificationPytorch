from config import config
import utils.files.pathutils as pathutils
# Set up system path for relative imports
pathutils.setup_sys_path()
from utils.evaluation.modelevaluator import ModelEvaluator
from utils.logging.loggerfactory import LoggerFactory
from utils.tensorboard.tensorboardwriter import TensorBoardWriter
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
    #valid_test_loader = datasetutils.get_data_loader_by_name("valid+test", config=this_config, shuffle=True)

    # intialize the model
    with ModelEvaluator.from_file(device, TensorBoardWriter(config=this_config), this_config) as modelEvaluator:
        prediction_results = modelEvaluator.predict(test_loader)
        test_predictions, test_correct_labels, test_loss = prediction_results['predictions'], prediction_results['true_labels'], prediction_results['avg_loss']
        f1, precision, recall = modelEvaluator.evaluate_predictions(test_loader, test_predictions, test_correct_labels, modelEvaluator.epochs, 'Test', 'micro', 'Test', 0.5)

        logger.info(f"Test Loss: {test_loss}, F1: {f1}, Precision: {precision}, Recall: {recall}")


def main():
    # Call the train_model function with the configuration object
    evaluate_model(config)

if __name__ == '__main__':
    main()