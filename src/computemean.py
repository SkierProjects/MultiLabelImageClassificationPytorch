import imclaslib.files.pathutils as pathutils
import imclaslib.dataset.datasetutils as datasetutils
import torch
from imclaslib.logging.loggerfactory import LoggerFactory
from imclaslib.config import Config

config = Config("default_config.yml")
# Set up logging for the training process
logger = LoggerFactory.setup_logging("logger", config, log_file=pathutils.combine_path(config, 
    pathutils.get_log_dir_path(config), 
    f"CalculateDatasetMeanStd",
    f"{pathutils.get_datetime()}.log"))

def compute_mean_std(dataloader):
    channels_sum, channels_squared_sum, total_images = 0, 0, 0

    for data in dataloader:
        images = data.get('image')

        if images is None:
            # Skip corrupted or missing images
            continue

        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected images to be a torch.Tensor but got {type(images)}")
        if not images.is_floating_point():
            images = images.float()  # Convert images to float if they're not already

        # Rearrange batch to be the shape of [B, C, W * H]
        images = images.view(images.size(0), images.size(1), -1)
        
        # Update total sum and squared sum
        channels_sum += images.mean(dim=[0, 2]) * images.size(0)
        channels_squared_sum += (images ** 2).mean(dim=[0, 2]) * images.size(0)
        total_images += images.size(0)

    # Compute mean and std
    mean = channels_sum / total_images
    std = (channels_squared_sum / total_images - mean ** 2) ** 0.5

    return mean, std
if __name__ == '__main__':
    # DataLoader for your dataset
    config.dataset_normalization_mean = None
    config.dataset_normalization_std = None
    dataloader = datasetutils.get_data_loader_by_name('all', config, shuffle=True)

    try:
        # Calculate mean and std
        mean, std = compute_mean_std(dataloader)
        logger.info(f'Mean: {mean}')
        logger.info(f'Std: {std}')
    except Exception as e:
        print(f'An error occurred during computation: {e}')