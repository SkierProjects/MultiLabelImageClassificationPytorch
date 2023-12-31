import torch
import hashlib
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.config import config
from src.utils.logging.loggerfactory import LoggerFactory
import pandas as pd
logger = LoggerFactory.get_logger(f"logger.{__name__}")

class ImageDataset(Dataset):
    """
    A dataset class for loading and transforming images for model training and evaluation.
    """

    def __init__(self, csv, mode, random_state=42, config=config): 
        """
        Initializes the dataset with images and labels based on the provided CSV file and mode.
        
        Parameters:
        - csv: pandas.DataFrame, contains file paths to images and associated labels.
        - mode: str, one of 'train', 'valid', or 'test' to determine dataset usage.
        - config: Config, configuration object containing dataset parameters.
        - random_state: int, random state for reproducible train-test splits.
        """
        if mode not in ['train', 'valid', 'test', 'valid+test']:
            raise ValueError("Mode must be 'train', 'valid', 'test', or 'valid+test'.")
        
        self.csv = csv
        self.config = config
        self.mode = mode
        self.all_image_names = self.csv[:]['filepath']
        self.all_labels = np.array(self.csv.drop(['filepath'], axis=1))
        self.image_size = self.config.image_size
        self.csv['identifier'] = self.csv['filepath'].apply(lambda x: x.split('/')[-1])
        train_size = config.train_percentage
        valid_size = config.valid_percentage
        test_size = config.test_percentage
        total_size = train_size + valid_size + test_size
        if total_size > 100:
            raise ValueError("The sum of train, valid, and test percentages should be <= 100.")

        # Convert self.all_image_names to a list if it's a pandas Series
        self.all_image_names = self.all_image_names.tolist()
        
        # Perform a stable split
        train_data, valid_data, test_data = stable_split(
            self.csv, train_size, valid_size, test_size, random_state=random_state
        )

        # Map back to the original data format
        train_names = train_data['filepath'].tolist()
        train_labels = np.array(train_data.drop(['filepath', 'identifier'], axis=1))
        
        valid_names = valid_data['filepath'].tolist()
        valid_labels = np.array(valid_data.drop(['filepath', 'identifier'], axis=1))

        test_names = test_data['filepath'].tolist()
        test_labels = np.array(test_data.drop(['filepath', 'identifier'], axis=1))

        # Concatenate validation and test sets to create valid+test set
        valid_test_names = np.concatenate((valid_names, test_names))
        valid_test_labels = np.vstack((valid_labels, test_labels))

        # Assign data based on mode
        if self.mode == 'train':
            self.image_names = train_names
            self.labels = train_labels
            self.transform = self.train_transforms()
        elif self.mode == 'valid':
            self.image_names = valid_names
            self.labels = valid_labels
            self.transform = self.valid_transforms()
        elif self.mode == 'test':
            self.image_names = test_names
            self.labels = test_labels
            self.transform = self.test_transforms()
        elif self.mode == 'valid+test':
            # Combine valid and test sets for the valid+test mode
            self.image_names = valid_test_names
            self.labels = valid_test_labels
            self.transform = self.test_transforms()
        else:
            raise ValueError("Mode must be 'train', 'valid', 'test', or 'valid+test'.")

    def train_transforms(self):
    
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),  # Moved after ToTensor
            transforms.Normalize(mean=self.config.dataset_normalization_mean, std=self.config.dataset_normalization_std),
        ])

    def valid_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.dataset_normalization_mean, std=self.config.dataset_normalization_std),
        ])

    def test_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.dataset_normalization_mean, std=self.config.dataset_normalization_std),
        ])

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.image_names)
    
    def __getitem__(self, index):
        """
        Retrieves an image and its labels at the given index, applying appropriate transforms.
        """
        image_path = self.image_names[index]
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Warning: Image not found or corrupted at path: {image_path}")
            return None
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]
        
        return {
            'image': image,
            'label': torch.tensor(targets, dtype=torch.float32),
            'image_path': image_path
        }
    
def stable_hash(x):
    # Use a large prime number to take the modulus of the hash
    large_prime = 2**61 - 1
    return int(hashlib.sha256(x.encode('utf-8')).hexdigest(), 16) % large_prime

def stable_split(data, train_percent, valid_percent, test_percent, random_state=None):
    # Ensure that the sum of the sizes is <= 1
    if train_percent + valid_percent + test_percent > 100:
        raise ValueError("The sum of train, valid, and test sizes should be <= 100.")

    if random_state is not None:
        np.random.seed(random_state)  # Set random seed for reproducibility

    # Assign a unique number to each element based on a hash of its identifier
    hashed_ids = data['identifier'].apply(stable_hash)

    # Calculate the split thresholds
    train_threshold = np.percentile(hashed_ids, train_percent)
    valid_threshold = np.percentile(hashed_ids, (train_percent + valid_percent))

    # Determine the subset for each element based on its hashed ID
    train_mask = hashed_ids < train_threshold
    valid_mask = (hashed_ids >= train_threshold) & (hashed_ids < valid_threshold)
    test_mask = hashed_ids >= valid_threshold

    train_data = data[train_mask]
    valid_data = data[valid_mask]
    test_data = data[test_mask]

    return train_data, valid_data, test_data