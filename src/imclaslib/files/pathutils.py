from pathlib import Path
from datetime import datetime
import sys

def get_best_model_path(config):
    """
    Gets the path for the best model's checkpoint file.

    Returns:
        Path: The path for the best model's checkpoint file.
    """
    return combine_path(config, get_output_dir_path(config), "best_model.pth")

def get_model_to_load_path(config):
    """
    Gets the path for the model to load (defined in the config)

    Parameters:
        config (object): Configuration object containing the model to load name.

    Returns:
        Path: The path for the model to loads checkpoint file.
    """
    return combine_path(config, get_output_dir_path(config), f"{config.model_name_to_load}.pth")

def get_log_dir_path(config):
    """
    Gets the path for the project's log directory, creating it if it doesn't exist.

    Returns:
        Path: The log directory path.
    """
    return combine_path(config, config.logs_folder)


def get_tensorboard_log_dir_path(config):
    """
    Gets the path for the TensorBoard log directory.

    Returns:
        Path: The TensorBoard log directory path.
    """
    return combine_path(config, config.logs_tensorboard_folder)

def get_output_dir_path(config):
    """
    Gets the path for the project's output directory.

    Returns:
        Path: The output directory path.
    """
    return combine_path(config, config.model_folder)

def combine_path(config, *args):
    """
    Combines multiple path components into a single Path object.

    Parameters:
        *args: A variable number of path components.

    Returns:
        Path: The combined path.
    """
    raw_path = Path(*args)
    if config.using_wsl:
        raw_path = convert_windows_path_to_wsl(raw_path)
    return raw_path

def get_datetime():
    """
    Gets the current date and time in the format YYYYMMDD_HHMMSS.

    Returns:
        str: The current date and time as a string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def convert_windows_path_to_wsl(windows_path):
    """
    Convert a Windows file path to a WSL file path.
    """
    input_is_path = isinstance(windows_path, Path)
    windows_path_str = str(windows_path)

    # Replace the drive letter with '/mnt/' and lower the case of the drive letter
    if windows_path_str[1:3] == ':\\':
        wsl_path = '/mnt/' + windows_path_str[0].lower() + windows_path_str[2:]
    else:
        wsl_path = windows_path_str  # If the path is already in the correct format

    # Replace backslashes with forward slashes
    wsl_path = wsl_path.replace('\\', '/')
    
    return Path(wsl_path) if input_is_path else wsl_path

def convert_wsl_path_to_windows(wsl_path):
    """
    Convert a WSL file path to a Windows file path.
    """
    input_is_path = isinstance(wsl_path, Path)
    wsl_path_str = str(wsl_path)

    # Check if the path starts with '/mnt/' (case-insensitive)
    if wsl_path_str.lower().startswith('/mnt/'):
        # Extract the drive letter and construct the Windows path
        drive_letter = wsl_path_str[5]
        windows_path = f"{drive_letter.upper()}:{wsl_path_str[6:]}"
    else:
        windows_path = wsl_path_str  # If the path is already in the correct format

    # Replace forward slashes with backslashes
    windows_path = windows_path.replace('/', '\\')
    
    return Path(windows_path) if input_is_path else windows_path

def get_dataset_path(config):
    """
    Gets the path for the dataset file based on the configuration.

    Parameters:
        config (object): Configuration object containing the dataset file name.

    Returns:
        Path: The dataset file path.
    """
    return combine_path(config, config.dataset_path)

def get_train_many_models_file(config):
    """
    Gets the path for the JSON file containing configurations for training many models.

    Returns:
        Path: The path to the 'train_many_models.json' file.
    """
    return combine_path(config, config.train_many_models_path)

def get_test_many_models_file(config):
    """
    Gets the path for the JSON file containing configurations for training many models.

    Returns:
        Path: The path to the 'train_many_models.json' file.
    """
    return combine_path(config, config.test_many_models_path)

def get_tags_path(config):
    """
    Gets the path for the tags file containing all possible tags.

    Returns:
        Path: The path to the 'tags.text' file.
    """
    return combine_path(config, config.model_tags_path)

def get_graph_path(config):
    return combine_path(config, config.model_gcn_graph_path)
