from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from imclaslib.logging.loggerfactory import LoggerFactory
logger = LoggerFactory.get_logger(f"logger.{__name__}")

class VideoDatasetPredict(Dataset):
    """Custom dataset for loading images from a list of image paths."""
    def __init__(self, video_path, time_interval, config):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = int(self.fps * time_interval)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.image_paths = video_path
        self.config = config
        self.transform = VideoDatasetPredict.test_transforms(self.config)

    def __len__(self):
        return self.total_frames // self.frame_interval
    
    def __del__(self):
        # Release the video capture object
        self.cap.release()
    
    @staticmethod
    def test_transforms(config):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.model_image_size, config.model_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.dataset_normalization_mean, std=config.dataset_normalization_std),
        ])

    def __getitem__(self, idx):
        # Calculate the actual frame index
        frame_idx = idx * self.frame_interval

        # Set the video capture to the correct frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        # Check if the frame was read correctly
        if not ret:
            logger.warn(f"Frame at index {frame_idx} could not be read")
            return None

        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = self.transform(frame)

        return {
            'image': frame,
            'frame_count': frame_idx
        }
    
    
