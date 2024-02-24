from torch.utils.data import Dataset
import torch
from torchvision.transforms import v2 as V2
import cv2
from imclaslib.logging.loggerfactory import LoggerFactory
from PIL import Image
logger = LoggerFactory.get_logger(f"logger.{__name__}")

class ImageDatasetPredict(Dataset):
    """Custom dataset for loading images from a list of image paths."""
    def __init__(self, image_paths, config):
        self.image_paths = image_paths
        self.config = config
        self.preprocess_fn = V2.Compose([
            V2.ToImage(),
            V2.Resize((config.model_image_size, config.model_image_size)),
            V2.ToDtype(torch.float32, scale=True),
            V2.Normalize(mean=config.dataset_normalization_mean, std=config.dataset_normalization_std),
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def preprocess_single_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            image = None
        if image is None:
            logger.warning(f"Warning: Image not found or corrupted at path: {image_path}")
            return None
        # apply image transforms
        image = self.preprocess_fn(image)
        return image

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.preprocess_single_image(image_path)

        if image is None:
            # Log that we're using a placeholder for a specific image
            #logger.warning(f"Using placeholder for missing or corrupted image at path: {image_path}")
            
            # Create a placeholder tensor (e.g., a tensor of zeros)
            # The shape should match your model's input size, e.g., (C, H, W)
            C, H, W = 3, self.config.model_image_size, self.config.model_image_size
            image = torch.zeros((C, H, W), dtype=torch.float32)
            image_path = "INVALID:" + image_path
        
        return {
            'image': image,
            'image_path': image_path
        }
    
