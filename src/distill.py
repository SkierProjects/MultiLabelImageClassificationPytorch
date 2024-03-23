from imclaslib.config import Config
import imclaslib.files.pathutils as pathutils
from imclaslib.training.distill_model import distill_model
from imclaslib.logging.loggerfactory import LoggerFactory
from imclaslib.wandb.wandb_writer import WandbWriter

# Set up logging for the training process
config = Config("default_config.yml")
#TODO: CLEAN UP THIS. ITS REUSED IN A TON OF PLACES
logger = LoggerFactory.setup_logging("logger", config, log_file=pathutils.combine_path(config, 
    pathutils.get_log_dir_path(config), 
    f"{config.model_name}_{config.model_image_size}_{config.model_weights}",
    f"distill__{pathutils.get_datetime()}.log"))


def main(json_file_path):
    """
    Train multiple models as per configurations provided in the JSON file.
    
    Parameters:
    - json_file_path: str, the path to the JSON file containing the configurations
    """
    configs = Config.load_configs_from_file(json_file_path, config)
    teacher_config = configs[0]
    student_config = configs[1]
    logger.info(teacher_config.model_name)
    logger.info(student_config.model_name)
    logger.info(f"Starting distillation for model: {student_config.model_name}, image size: {student_config.model_image_size}, dropout: {student_config.train_dropout_prob}, weights: {student_config.model_weights}, l2: {student_config.train_l2_enabled}, fp16: {student_config.model_fp16}, dataset version: {student_config.dataset_version}")
    with WandbWriter(student_config) as wandb_writer:
        try:
            distill_model(teacher_config, student_config, wandb_writer)
        except Exception as e:
            logger.error('Error at %s', 'distill', exc_info=e)
            raise e

if __name__ == '__main__':
    # Get the path to the JSON file containing the model configurations
    json_file_path = pathutils.get_distill_models_file(config)
    main(json_file_path)