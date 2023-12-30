from pathlib import Path
from datetime import datetime
import sys

def get_root_path():
    """
    Gets the project's root directory path.

    Returns:
        Path: The root directory path as a Path object.
    """
    return Path(__file__).resolve().parents[3]

def get_best_model_path():
    """
    Gets the path for the best model's checkpoint file.

    Returns:
        Path: The path for the best model's checkpoint file.
    """
    return Path(get_output_dir_path(), "best_model.pth")

def get_model_to_load_path(config):
    """
    Gets the path for the model to load (defined in the config)

    Parameters:
        config (object): Configuration object containing the model to load name.

    Returns:
        Path: The path for the model to loads checkpoint file.
    """
    return Path(get_output_dir_path(), f"{config.model_name_to_load}.pth")

def get_log_dir_path():
    """
    Gets the path for the project's log directory, creating it if it doesn't exist.

    Returns:
        Path: The log directory path.
    """
    log_directory = Path(get_root_path(), "logs")
    log_directory.mkdir(exist_ok=True)
    return log_directory

def get_tensorboard_log_dir_path():
    """
    Gets the path for the TensorBoard log directory.

    Returns:
        Path: The TensorBoard log directory path.
    """
    tensorboard_log_directory = Path(get_root_path(), "tensorboard_logs")
    return tensorboard_log_directory

def get_output_dir_path():
    """
    Gets the path for the project's output directory.

    Returns:
        Path: The output directory path.
    """
    output_directory = Path(get_root_path(), "outputs")
    return output_directory

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

def setup_sys_path():
    """
    Prepends the project's root directory to the system path.
    """
    rootpath = str(get_root_path())
    sys.path.insert(0, rootpath)

def get_dataset_path(config):
    """
    Gets the path for the dataset file based on the configuration.

    Parameters:
        config (object): Configuration object containing the dataset file name.

    Returns:
        Path: The dataset file path.
    """
    return Path(get_root_path(), "Dataset", config.dataset_file_name)

def get_many_models_json():
    """
    Gets the path for the JSON file containing configurations for training many models.

    Returns:
        Path: The path to the 'train_many_models.json' file.
    """
    return Path(get_root_path(), "src", "train_many_models.json")