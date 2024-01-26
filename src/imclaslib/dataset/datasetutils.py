import pandas as pd
from imclaslib.dataset.image_dataset import ImageDataset
from torch.utils.data import DataLoader
import imclaslib.files.pathutils as pathutils

# Global variable to cache the dataset CSV after being read for the first time.
dataset_csv = None

def get_train_valid_test_loaders(config):
    """
    Creates and returns DataLoaders for the training, validation, and test sets.

    Parameters:
    - config: An immutable configuration object with necessary parameters.

    Returns:
    - Tuple of DataLoaders: (train_loader, valid_loader, test_loader)
    """
    global dataset_csv
    dataset_csv = __get_dataset_csv(config)
    train_data = ImageDataset(dataset_csv, mode='train', config=config)
    valid_data = ImageDataset(dataset_csv, mode='valid', config=config)
    test_data = ImageDataset(dataset_csv, mode='test', config=config)

    train_loader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.train_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.train_batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def get_data_loader_by_name(mode, config, shuffle=False):
    """
    Creates and returns a DataLoader for the specified mode.

    Parameters:
    - mode: A string indicating the mode ('train', 'valid', 'test').
    - config: An immutable configuration object with necessary parameters.
    - shuffle: A boolean indicating whether to shuffle the dataset.

    Returns:
    - DataLoader for the specified mode.
    """
    global dataset_csv
    dataset_csv = __get_dataset_csv(config)
    data = ImageDataset(dataset_csv, mode=mode, config=config)
    loader = DataLoader(data, batch_size=config.test_batch_size, shuffle=shuffle)
    return loader

def get_dataset_tag_mappings(config):
    """
    Retrieves a mapping from index to tag names from the dataset CSV.

    Parameters:
    - config: An immutable configuration object with necessary parameters.

    Returns:
    - A dictionary mapping indices to tag names.
    """
    global dataset_csv
    dataset_csv = __get_dataset_csv(config)
    return __get_index_to_tag_mapping(dataset_csv)

def get_tag_to_index_mapping(config):
    """
    Retrieves a mapping from tag names to indices by reading from a text file.

    Parameters:
    - tags_txt_path: Path to the text file containing tags, one on each line.

    Returns:
    - A dictionary mapping tag names to indices.
    """
    tags_txt_path = pathutils.get_tags_path(config)
    tag_to_index = {}
    with open(tags_txt_path, 'r', encoding='utf-8') as file:
        for index, tag in enumerate(file):
            tag_to_index[tag.strip()] = index  # Remove any leading/trailing whitespace
    return tag_to_index

def get_index_to_tag_mapping(config):
    """
    Retrieves a mapping from indices to tag names by reading from a text file.

    Parameters:
    - tags_txt_path: Path to the text file containing tags, one on each line.

    Returns:
    - A dictionary mapping indices to tag names.
    """
    tags_txt_path = pathutils.get_tags_path(config)
    index_to_tag = {}
    with open(tags_txt_path, 'r', encoding='utf-8') as file:
        for index, tag in enumerate(file):
            index_to_tag[index] = tag.strip()  # Remove any leading/trailing whitespace
    return index_to_tag

def __get_index_to_tag_mapping(csv):
    """
    Helper function to create a mapping from column index to tag name.

    Parameters:
    - csv: The dataset CSV DataFrame.

    Returns:
    - A dictionary mapping indices to tag names.
    """
    tag_columns = csv.columns[1:]
    index_to_tag = {index: tag for index, tag in enumerate(tag_columns)}
    return index_to_tag

def __get_dataset_csv(config):
    """
    Retrieves the dataset CSV, reading it from file if not already cached.

    Parameters:
    - config: An immutable configuration object with necessary parameters.

    Returns:
    - The dataset CSV DataFrame.
    """
    global dataset_csv
    if dataset_csv is None:
        dataset_csv = pd.read_csv(pathutils.get_dataset_path(config))
    return dataset_csv
