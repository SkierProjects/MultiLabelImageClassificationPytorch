from pathlib import Path
from datetime import datetime
import sys

def get_best_model_path(config):
    """
    Gets the path for the best model's checkpoint file.

    Returns:
        Path: The path for the best model's checkpoint file.
    """
    return Path(get_output_dir_path(config), "best_model.pth")

def get_model_to_load_path(config):
    """
    Gets the path for the model to load (defined in the config)

    Parameters:
        config (object): Configuration object containing the model to load name.

    Returns:
        Path: The path for the model to loads checkpoint file.
    """
    return Path(get_output_dir_path(config), f"{config.model_name_to_load}.pth")

def get_log_dir_path(config):
    """
    Gets the path for the project's log directory, creating it if it doesn't exist.

    Returns:
        Path: The log directory path.
    """
    return Path(config.paths_log_folder)


def get_tensorboard_log_dir_path(config):
    """
    Gets the path for the TensorBoard log directory.

    Returns:
        Path: The TensorBoard log directory path.
    """
    return Path(config.paths_tensorboard_log_folder)

def get_output_dir_path(config):
    """
    Gets the path for the project's output directory.

    Returns:
        Path: The output directory path.
    """
    return Path(config.paths_output_folder)

def combine_path(*args):
    """
    Combines multiple path components into a single Path object.

    Parameters:
        *args: A variable number of path components.

    Returns:
        Path: The combined path.
    """
    return Path(*args)

def get_datetime():
    """
    Gets the current date and time in the format YYYYMMDD_HHMMSS.

    Returns:
        str: The current date and time as a string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_dataset_path(config):
    """
    Gets the path for the dataset file based on the configuration.

    Parameters:
        config (object): Configuration object containing the dataset file name.

    Returns:
        Path: The dataset file path.
    """
    return Path(config.paths_dataset)

def get_train_many_models_file(config):
    """
    Gets the path for the JSON file containing configurations for training many models.

    Returns:
        Path: The path to the 'train_many_models.json' file.
    """
    return Path(config.paths_train_many_models)

def get_test_many_models_file(config):
    """
    Gets the path for the JSON file containing configurations for training many models.

    Returns:
        Path: The path to the 'train_many_models.json' file.
    """
    return Path(config.paths_test_many_models)

def get_tags_path(config):
    """
    Gets the path for the tags file containing all possible tags.

    Returns:
        Path: The path to the 'tags.text' file.
    """
    return Path(config.paths_tags)

def get_graph_path(config):
    return Path(config.paths_graph)
