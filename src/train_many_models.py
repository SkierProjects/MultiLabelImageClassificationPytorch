from config import config
import utils.files.pathutils as pathutils

# Set up system path for relative imports
pathutils.setup_sys_path()

from utils.training.train_model import train_model
from utils.files import jsonutils
from src.utils.logging.loggerfactory import LoggerFactory

# Set up logging for the training process
logger = LoggerFactory.setup_logging("logger", log_file=pathutils.combine_path(
    pathutils.get_log_dir_path(), 
    f"{config.model_name}_{config.image_size}_{config.model_weights}",
    f"train__{pathutils.get_datetime()}.log"))


def main(json_file_path):
    """
    Train multiple models as per configurations provided in the JSON file.
    
    Parameters:
    - json_file_path: str, the path to the JSON file containing the configurations
    """
    configs = jsonutils.load_configs_from_json(json_file_path)
    for config_instance in configs:
        logger.info(f"Starting training for model: {config_instance.model_name}, image size: {config_instance.image_size}, dropout: {config_instance.model_dropout_prob}, weights: {config_instance.model_weights}")
        train_model(config_instance)

if __name__ == '__main__':
    # Get the path to the JSON file containing the model configurations
    json_file_path = pathutils.get_train_many_models_json()
    main(json_file_path)