from imclaslib.config import Config
import imclaslib.files.pathutils as pathutils
from imclaslib.evaluation.test_model import evaluate_model
from imclaslib.files.modelloadingutils import update_config_from_model_file
from imclaslib.logging.loggerfactory import LoggerFactory

# Set up logging for the training process
config = Config("default_config.yml")
logger = LoggerFactory.setup_logging("logger", config, log_file=pathutils.combine_path(
    pathutils.get_log_dir_path(config), 
    f"{config.model_name}_{config.model_image_size}_{config.model_weights}",
    f"train__{pathutils.get_datetime()}.log"))


def main(json_file_path):
    """
    Train multiple models as per configurations provided in the JSON file.
    
    Parameters:
    - json_file_path: str, the path to the JSON file containing the configurations
    """
    configs = Config.load_configs_from_file(json_file_path, config)
    for config_instance in configs:
        update_config_from_model_file(config_instance)
        logger.info(f"Starting Evaluating for model: {config_instance.model_name}, image size: {config_instance.model_image_size}, weights: {config_instance.model_weights}")
        try:
            evaluate_model(config_instance)
        except Exception as e:
            logger.error(f"Failed testing for model: {config_instance.model_name}, image size: {config_instance.model_image_size}, weights: {config_instance.model_weights} Inner:{e.strerror}")

if __name__ == '__main__':
    # Get the path to the JSON file containing the model configurations
    json_file_path = pathutils.get_test_many_models_file(config)
    main(json_file_path)