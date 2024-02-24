import os
import torch
import hashlib
import cv2
import numpy as np
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2.functional import resize
from torch.utils.data import Dataset
from imclaslib.files import pathutils
from imclaslib.logging.loggerfactory import LoggerFactory
import imclaslib.dataset.datasetutils as datasetutils
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import pandas as pd
logger = LoggerFactory.get_logger(f"logger.{__name__}")

class ImageDataset(Dataset):
    """
    A dataset class for loading and transforming images for model training and evaluation.
    """

    def __init__(self, csv, mode, config, random_state=42): 
        """
        Initializes the dataset with images and labels based on the provided CSV file and mode.
        
        Parameters:
        - csv: pandas.DataFrame, contains file paths to images and associated labels.
        - mode: str, one of 'train', 'valid', or 'test' to determine dataset usage.
        - config: Config, configuration object containing dataset parameters.
        - random_state: int, random state for reproducible train-test splits.
        """
        if mode not in ['train', 'valid', 'test', 'valid+test', 'all']:
            raise ValueError("Mode must be 'train', 'valid', 'test', 'valid+test', or 'all'.")
        
        self.label_mapping = config.dataset_tags_mapping_dict
        self.class_to_idx = datasetutils.get_tag_to_index_mapping(config)
        self.csv = csv
        self.config = config
        self.mode = mode
        #self.csv['identifier'] = self.csv['filepath']
        if config.using_wsl:
            self.csv['filepath'] = self.csv['filepath'].apply(pathutils.convert_windows_path_to_wsl)

        self.csv['identifier'] = self.csv['filepath'].apply(lambda x: os.path.basename(x))
        self.all_image_names = self.csv[:]['filepath']
        
        # Assuming all columns other than 'filepath' and 'identifier' are labels and should be numeric
        label_columns = self.csv.columns.drop(['filepath', 'identifier'])

        # Convert label columns to a numeric type (e.g., float32) and handle NaNs
        self.csv[label_columns] = self.csv[label_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)

        # Now create the all_labels array with a uniform dtype
        self.all_labels = np.array(self.csv[label_columns])

        self.image_size = self.config.model_image_size
        train_size = config.dataset_train_percentage
        valid_size = config.dataset_valid_percentage
        test_size = config.dataset_test_percentage
        total_size = train_size + valid_size + test_size
        if total_size > 100:
            raise ValueError("The sum of train, valid, and test percentages should be <= 100.")

        # Convert self.all_image_names to a list if it's a pandas Series
        self.all_image_names = self.all_image_names.tolist()
        
        # Perform a stable split
        train_data, valid_data, test_data = self.stable_split(
            self.csv, train_size, valid_size, test_size, random_state=random_state
        )

        # Map back to the original data format
        train_names = train_data['filepath'].tolist()
        train_labels = self.map_labels(train_data)
        
        valid_names = valid_data['filepath'].tolist()
        valid_labels = self.map_labels(valid_data)

        test_names = test_data['filepath'].tolist()
        test_labels = self.map_labels(test_data)

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
        elif self.mode == 'all':
            # Combine valid and test sets for the valid+test mode
            self.image_names = self.all_image_names
            self.labels = self.all_labels
            self.transform = self.test_transforms()
        else:
            raise ValueError("Mode must be 'train', 'valid', 'test', or 'valid+test'.")
        
        if self.config.dataset_preprocess_to_RAM:
            self.data = []
            for index, file_path in enumerate(self.image_names):
                label = self.labels[index]

                image = Image.open(file_path).convert('RGB')
                if image is None:
                    logger.warning(f"Warning: Image not found or corrupted at path: {file_path}")
                    return None
                image = resize(image, (self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC)
                item = {
                    'image': image,
                    'label': torch.tensor(label, dtype=torch.float32),
                    'image_path': file_path
                }
                self.data.append(item)

    # Apply the label mapping to each subset after splitting
    def map_labels(self, data):
        # Initialize a label matrix for the given subset of data
        label_matrix = np.zeros((len(data), len(self.class_to_idx)), dtype=float)
        
        # Map the old and new labels using the dictionary
        for col in self.csv.columns:
            if col in self.class_to_idx:
                # This label is directly in the class_to_idx, so use it as is
                class_idx = self.class_to_idx[col]
                label_matrix[:, class_idx] = data[col].values
            if col in self.label_mapping:
                # This label should be mapped to another label
                mapped_label = self.label_mapping[col]
                class_idx = self.class_to_idx[mapped_label]
                # Set the broader category label to true if this or any previously mapped label is true
                label_matrix[:, class_idx] = np.logical_or(label_matrix[:, class_idx], data[col].values)
        return label_matrix

    # Define a function to scale the augmentation parameters based on input level (0-10)
    def scale_parameter(self, min_val, max_val, level):
        """Scales the parameter based on the augmentation level (0-10)."""
        return min_val + (max_val - min_val) * level / 10

    def train_transforms(self):
        augmentation_level = self.config.dataset_augmentation_level
        assert 0 <= augmentation_level <= 10, "Augmentation level must be between 0 and 10"

        # Define the augmentation parameters scaled by the augmentation_level
        horizontal_flip_prob = self.scale_parameter(0, 0.5, augmentation_level)
        color_jitter_brightness = self.scale_parameter(0, 0.5, augmentation_level)
        color_jitter_contrast = self.scale_parameter(0, 0.5, augmentation_level)
        color_jitter_saturation = self.scale_parameter(0, 0.5, augmentation_level)
        rotation_degrees = self.scale_parameter(0, 45, augmentation_level)
        affine_transform_degrees = self.scale_parameter(0, 10, augmentation_level)
        affine_transform_translate = self.scale_parameter(0, 0.05, augmentation_level)
        affine_transform_scale_min = self.scale_parameter(1, 0.95, augmentation_level)
        affine_transform_scale_max = self.scale_parameter(1, 1.05, augmentation_level)
        perspective_distortion_scale = self.scale_parameter(0, 0.2, augmentation_level)
        gaussian_blur_sigma = self.scale_parameter(0.1, 2, augmentation_level)
        random_erasing_prob = self.scale_parameter(0, 0.3, augmentation_level)

        # Now, create the list of transforms with the scaled parameters
        transforms_list = []
        

        if self.config.dataset_normalization_mean == None:
            transforms_list = [
                transforms.ToImage(),
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob) if augmentation_level > 0 else None,
                transforms.ColorJitter(brightness=color_jitter_brightness, contrast=color_jitter_contrast, saturation=color_jitter_saturation) if augmentation_level > 0 else None,
                transforms.RandomRotation(degrees=rotation_degrees) if augmentation_level > 0 else None,
                transforms.RandomAffine(degrees=affine_transform_degrees, translate=(affine_transform_translate, affine_transform_translate),
                                        scale=(affine_transform_scale_min, affine_transform_scale_max)) if augmentation_level > 0 else None,
                transforms.RandomPerspective(distortion_scale=perspective_distortion_scale, p=0.5) if augmentation_level > 0 else None,
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=gaussian_blur_sigma) if augmentation_level > 0 else None,
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0) if augmentation_level > 0 else None,
            ]
        elif self.config.dataset_preprocess_to_RAM:
            transforms_list = [
                transforms.ToImage(),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob) if augmentation_level > 0 else None,
                transforms.ColorJitter(brightness=color_jitter_brightness, contrast=color_jitter_contrast, saturation=color_jitter_saturation) if augmentation_level > 0 else None,
                transforms.RandomRotation(degrees=rotation_degrees) if augmentation_level > 0 else None,
                transforms.RandomAffine(degrees=affine_transform_degrees, translate=(affine_transform_translate, affine_transform_translate),
                                        scale=(affine_transform_scale_min, affine_transform_scale_max)) if augmentation_level > 0 else None,
                transforms.RandomPerspective(distortion_scale=perspective_distortion_scale, p=0.5) if augmentation_level > 0 else None,
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=gaussian_blur_sigma) if augmentation_level > 0 else None,
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0) if augmentation_level > 0 else None,
                transforms.Normalize(mean=self.config.dataset_normalization_mean, std=self.config.dataset_normalization_std),
            ]
        else:
            transforms_list = [
                transforms.ToImage(),
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob) if augmentation_level > 0 else None,
                transforms.ColorJitter(brightness=color_jitter_brightness, contrast=color_jitter_contrast, saturation=color_jitter_saturation) if augmentation_level > 0 else None,
                transforms.RandomRotation(degrees=rotation_degrees) if augmentation_level > 0 else None,
                transforms.RandomAffine(degrees=affine_transform_degrees, translate=(affine_transform_translate, affine_transform_translate),
                                        scale=(affine_transform_scale_min, affine_transform_scale_max)) if augmentation_level > 0 else None,
                transforms.RandomPerspective(distortion_scale=perspective_distortion_scale, p=0.5) if augmentation_level > 0 else None,
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=gaussian_blur_sigma) if augmentation_level > 0 else None,
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0) if augmentation_level > 0 else None,
                transforms.Normalize(mean=self.config.dataset_normalization_mean, std=self.config.dataset_normalization_std),
            ]

        # Filter out None transforms (i.e., when augmentation_level is 0)
        transforms_list = [t for t in transforms_list if t is not None]

        return transforms.Compose(transforms_list)

    def valid_transforms(self):
        transforms_list = []
        if self.config.dataset_normalization_mean == None:
            transforms_list = [
                transforms.ToImage(),
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        elif self.config.dataset_preprocess_to_RAM:
            transforms_list = [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=self.config.dataset_normalization_mean, std=self.config.dataset_normalization_std),
            ]
        else:
            transforms_list = [
                transforms.ToImage(),
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=self.config.dataset_normalization_mean, std=self.config.dataset_normalization_std),
            ]
        return transforms.Compose(transforms_list)

    def test_transforms(self):
        transforms_list = []
        if self.config.dataset_normalization_mean == None:
            transforms_list = [
                transforms.ToImage(),
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        elif self.config.dataset_preprocess_to_RAM:
            transforms_list = [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=self.config.dataset_normalization_mean, std=self.config.dataset_normalization_std),
            ]
        else:
            transforms_list = [
                transforms.ToImage(),
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=self.config.dataset_normalization_mean, std=self.config.dataset_normalization_std),
            ]
            
        return transforms.Compose(transforms_list)

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.image_names)
    
    def __getitem__(self, index):
        """
        Retrieves an image and its labels at the given index, applying appropriate transforms.
        """
        if self.config.dataset_preprocess_to_RAM:
            return {
                'image': self.transform((self.data[index])['image']),
                'label': (self.data[index])['label'],
                'image_path': (self.data[index])['image_path']
            }
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
    
    def stable_hash(self, x):
        # Use a large prime number to take the modulus of the hash
        large_prime = 2**61 - 1
        return int(hashlib.sha256(x.encode('utf-8')).hexdigest(), 16) % large_prime

    def stable_split(self, data, train_percent, valid_percent, test_percent, random_state=None):
        # Ensure that the sum of the sizes is <= 1
        if train_percent + valid_percent + test_percent > 100:
            raise ValueError("The sum of train, valid, and test sizes should be <= 100.")

        if random_state is not None:
            np.random.seed(random_state)  # Set random seed for reproducibility

        # Assign a unique number to each element based on a hash of its identifier
        hashed_ids = data['identifier'].apply(lambda x: self.stable_hash(video_frame_group(x)))

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

def is_video_frame(identifier):
    return 'video' in identifier and 'frame' in identifier and 'studio' in identifier

def video_frame_group(identifier):
    if is_video_frame(identifier):
        splits = identifier.split('-', maxsplit=1)  
        return splits[0] + '-' + splits[1] # Returns 'studio_<id>-video_<id>'
    return identifier

