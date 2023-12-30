import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.config import config
from src.utils.logging.loggerfactory import LoggerFactory
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
        if mode not in ['train', 'valid', 'test']:
            raise ValueError("Mode must be 'train', 'valid', or 'test'.")
        
        self.csv = csv
        self.config = config
        self.mode = mode
        self.all_image_names = self.csv[:]['filepath']
        self.all_labels = np.array(self.csv.drop(['filepath'], axis=1))
        self.image_size = self.config.image_size

        # Convert self.all_image_names to a list if it's a pandas Series
        self.all_image_names = self.all_image_names.tolist()

        # Shuffle and split the data
        train_names, test_names, train_labels, test_labels = train_test_split(
            self.all_image_names, self.all_labels, test_size=0.2, random_state=random_state)

        # Further split the test set into validation and test sets
        test_names, valid_names, test_labels, valid_labels = train_test_split(
            test_names, test_labels, test_size=0.5, random_state=random_state)
        
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
            transforms.ToTensor(),
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
            'label': torch.tensor(targets, dtype=torch.float32)
        }