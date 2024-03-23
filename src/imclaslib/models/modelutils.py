from torch.optim.lr_scheduler import ReduceLROnPlateau
from imclaslib.logging.loggerfactory import LoggerFactory
logger = LoggerFactory.get_logger(f"logger.{__name__}")

def get_learningRate_scheduler(optimizer, config):
    """
    Creates a Learning Rate Scheduler to reduce learning rate during training

    Parameters:
        optimizer (torch.optim.Optimizer): The optimizer instance to load the state into.
        config: (Config): Configuration object containing dataset parameters.

    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: The learning rate reducer
    """
    return ReduceLROnPlateau(optimizer, mode='max', factor=config.train_learningrate_reducer_factor, patience=config.train_learningrate_reducer_patience, threshold=config.train_learningrate_reducer_threshold, min_lr=config.train_learningrate_reducer_min_lr)