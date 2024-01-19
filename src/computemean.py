import imclaslib.files.pathutils as pathutils

# Set up system path for relative imports
pathutils.setup_sys_path()

import imclaslib.dataset.datasetutils as datasetutils
import torch
from imclaslib.logging.loggerfactory import LoggerFactory
from imclaslib.config import Config

config = Config("default_config.yml")
# Set up logging for the training process
logger = LoggerFactory.setup_logging("logger", config, log_file=pathutils.combine_path(
    pathutils.get_log_dir_path(), 
    f"CalculateDatasetMeanStd",
    f"{pathutils.get_datetime()}.log"))

def compute_mean_std(dataloader):
    """
    Compute the mean and standard deviation of images in the given DataLoader.

    Parameters:
    - dataloader: DataLoader, a DataLoader containing the dataset to compute statistics on.

    Returns:
    - mean: torch.Tensor, the mean value for each channel across the dataset.
    - std: torch.Tensor, the standard deviation for each channel across the dataset.
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data in dataloader:
        # Get the image and its corresponding label from the data dictionary
        images = data['image']

        # Ensure that images is a torch tensor
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected images to be a torch.Tensor but got {type(images)}")

        # Rearrange batch to be the shape of [B, C, W * H]
        images = images.view(images.size(0), images.size(1), -1)
        
        # Update total sum and squared sum
        channels_sum += images.mean(dim=[0, 2])
        channels_squared_sum += (images ** 2).mean(dim=[0, 2])
        num_batches += 1

    # Compute mean and std
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
if __name__ == '__main__':
    # DataLoader for your dataset
    dataloader = datasetutils.get_data_loader_by_name('train', Config, shuffle=True)

    try:
        # Calculate mean and std
        mean, std = compute_mean_std(dataloader)
        logger.info(f'Mean: {mean}')
        logger.info(f'Std: {std}')
    except Exception as e:
        print(f'An error occurred during computation: {e}')