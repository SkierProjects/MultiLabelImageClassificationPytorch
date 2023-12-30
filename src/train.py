from config import config
import utils.files.pathutils as pathutils

# Set up system path for relative imports
pathutils.setup_sys_path()

from utils.training.train_model import train_model
from src.utils.logging.loggerfactory import LoggerFactory

# Set up logging for the training process
logger = LoggerFactory.setup_logging("logger", log_file=pathutils.combine_path(
    pathutils.get_log_dir_path(), 
    f"{config.model_name}_{config.image_size}_{config.model_weights}",
    f"train__{pathutils.get_datetime()}.log"))

def main():
    # Call the train_model function with the configuration object
    train_model(config)

if __name__ == '__main__':
    main()