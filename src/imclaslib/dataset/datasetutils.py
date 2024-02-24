import csv
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

    train_loader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True, num_workers=6, persistent_workers=True, pin_memory=False)
    valid_loader = DataLoader(valid_data, batch_size=config.train_batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False)
    test_loader = DataLoader(test_data, batch_size=config.train_batch_size, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False)

    return train_loader, valid_loader, test_loader

def get_data_loader_by_name(mode, config, shuffle=False, num_workers=1):
    """
    Creates and returns a DataLoader for the specified mode.

    Parameters:
    - mode: A string indicating the mode ('train', 'valid', 'test', or 'all').
    - config: An immutable configuration object with necessary parameters.
    - shuffle: A boolean indicating whether to shuffle the dataset.

    Returns:
    - DataLoader for the specified mode.
    """
    global dataset_csv
    dataset_csv = __get_dataset_csv(config)
    data = ImageDataset(dataset_csv, mode=mode, config=config)
    loader = DataLoader(data, batch_size=config.test_batch_size, shuffle=shuffle, pin_memory=False, persistent_workers=False, num_workers=num_workers)
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

def analyze_csv(config):

    csv_file_path = pathutils.get_dataset_path(config)
    # Initialize dictionaries to store annotation counts and file counts
    annotation_counts = {}
    file_counts = {'with_annotations': 0, 'without_annotations': 0}

    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Iterate through each row in the CSV
        for row in reader:
            file_name = row['filepath']
            
            # Count files without any annotations
            if all(value == '0' for key, value in row.items() if key != 'filepath'):
                file_counts['without_annotations'] += 1
            else:
                file_counts['with_annotations'] += 1
            
            # Count the usage of each annotation
            for annotation_name, annotation_value in row.items():
                if annotation_name != 'filepath':
                    annotation_counts[annotation_name] = annotation_counts.get(annotation_name, 0) + int(annotation_value)

    return annotation_counts, file_counts

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
