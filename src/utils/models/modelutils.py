from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.logging.loggerfactory import LoggerFactory
from src.config import config
logger = LoggerFactory.get_logger(f"logger.{__name__}")

def get_learningRate_scheduler(optimizer, config=config):
    """
    Creates a Learning Rate Scheduler to reduce learning rate during training

    Parameters:
        optimizer (torch.optim.Optimizer): The optimizer instance to load the state into.
        config: (Config): Configuration object containing dataset parameters.

    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: The learning rate reducer
    """
    return ReduceLROnPlateau(optimizer, mode='min', factor=config.learningrate_reducer_factor, patience=config.learningrate_reducer_patience, threshold=config.learningrate_reducer_threshold, verbose=True, min_lr=config.learningrate_reducer_min_lr)