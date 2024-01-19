from imclaslib.config import Config
import imclaslib.files.pathutils as pathutils
from imclaslib.training.train_model import train_model
from imclaslib.logging.loggerfactory import LoggerFactory

# Set up logging for the training process
config = Config("default_config.yml")
#TODO: CLEAN UP THIS. ITS REUSED IN A TON OF PLACES
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
        logger.info(f"Starting training for model: {config_instance.model_name}, image size: {config_instance.model_image_size}, dropout: {config_instance.model_dropout_prob}, weights: {config_instance.model_weights}")
        train_model(config_instance)

if __name__ == '__main__':
    # Get the path to the JSON file containing the model configurations
    json_file_path = pathutils.get_train_many_models_file(config)
    main(json_file_path)