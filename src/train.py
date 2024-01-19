from imclaslib.config import Config
import imclaslib.files.pathutils as pathutils
from imclaslib.training.train_model import train_model
from imclaslib.logging.loggerfactory import LoggerFactory

config = Config("default_config.yml")
# Set up logging for the training process
logger = LoggerFactory.setup_logging("logger", log_file=pathutils.combine_path(
    pathutils.get_log_dir_path(config), 
    f"{config.model_name}_{config.model_image_size}_{config.model_weights}",
    f"train__{pathutils.get_datetime()}.log"), config=config)

def main():
    # Call the train_model function with the configuration object
    train_model(config)

if __name__ == '__main__':
    main()