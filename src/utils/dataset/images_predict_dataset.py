from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from utils.logging.loggerfactory import LoggerFactory
logger = LoggerFactory.get_logger(f"logger.{__name__}")

class ImageDatasetPredict(Dataset):
    """Custom dataset for loading images from a list of image paths."""
    def __init__(self, image_paths, config):
        self.image_paths = image_paths
        self.config = config
        self.preprocess_fn = ImageDatasetPredict.test_transforms(self.config)

    def __len__(self):
        return len(self.image_paths)
    
    @staticmethod
    def test_transforms(config):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.dataset_normalization_mean, std=config.dataset_normalization_std),
        ])
    
    @staticmethod
    def preprocess_single_image(image_path, config):
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Warning: Image not found or corrupted at path: {image_path}")
            return None
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        transform = ImageDatasetPredict.test_transforms(config)
        image = transform(image)
        return image

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = ImageDatasetPredict.preprocess_single_image(image_path, self.config)
        return {
            'image': image,
            'image_path': image_path
        }
    
