from imclaslib.config import Config
import imclaslib.files.pathutils as pathutils
from imclaslib.evaluation.test_model import evaluate_model
from imclaslib.logging.loggerfactory import LoggerFactory
from imclaslib.config import Config
from imclaslib.wandb.wandb_writer import WandbWriter
# Set up logging for the training process
config = Config("default_config.yml")
logger = LoggerFactory.setup_logging("logger", log_file=pathutils.combine_path(config, 
    pathutils.get_log_dir_path(config), 
    f"{config.model_name}_{config.model_image_size}_{config.model_weights}",
    f"train__{pathutils.get_datetime()}.log"), config=config)


def main():
    # Call the train_model function with the configuration object
    with WandbWriter(config) as wandb_writer:
        evaluate_model(config, wandbWriter=wandb_writer)

if __name__ == '__main__':
    main()