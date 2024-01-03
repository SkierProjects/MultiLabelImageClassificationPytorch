import json
from src.config import config


def load_configs_from_json(json_file_path):
    """
    Load model configurations from a JSON file.
    
    Parameters:
    - json_file_path: str, the path to the JSON file containing the configurations
    
    Returns:
    - list, the list of configuration objects
    """
    with open(json_file_path, 'r') as f:
        configs_json = json.load(f)
    return [config.load_from_json(config_data) for config_data in configs_json]