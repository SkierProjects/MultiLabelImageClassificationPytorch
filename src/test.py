from imclaslib.config import Config

import imclaslib.files.pathutils as pathutils
# Set up system path for relative imports
pathutils.setup_sys_path()
from imclaslib.evaluation.test_model import evaluate_model
from imclaslib.logging.loggerfactory import LoggerFactory
from imclaslib.config import Config
# Set up logging for the training process
config = Config(str(pathutils.combine_path(pathutils.get_root_path(), "src", "default_config.yml")))
print(config.model_name)
logger = LoggerFactory.setup_logging("logger", log_file=pathutils.combine_path(
    pathutils.get_log_dir_path(), 
    f"{config.model_name}_{config.model_image_size}_{config.model_weights}",
    f"train__{pathutils.get_datetime()}.log"), config=config)


def main():
    # Call the train_model function with the configuration object
    print(config.model_name)
    #evaluate_model(config)

if __name__ == '__main__':
    main()