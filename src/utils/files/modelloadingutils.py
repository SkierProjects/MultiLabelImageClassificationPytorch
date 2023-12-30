import torch
from src.utils.logging.loggerfactory import LoggerFactory
import utils.files.pathutils as pathutils
import os

logger = LoggerFactory.get_logger(f"logger.{__name__}")

def save_best_model(model_state):
    """
    Saves the best model state to the predetermined best model path.

    Parameters:
        model_state (dict): State dictionary of the model to be saved.
    """
    torch.save(model_state, pathutils.get_best_model_path())

def save_final_model(model_state, f1_score, config):
    """
    Saves the final model state to a filename that includes model details such as name, image size, and F1 score.

    Parameters:
        model_state (dict): State dictionary of the model to be saved.
        f1_score (float): The F1 score of the model.
        config (object): Configuration object containing model_name and image_size.
    """
    final_model_path_template = os.path.join(str(pathutils.get_output_dir_path()), '{model_name}_{image_size}_{f1_score:.4f}.pth')
    final_model_path = final_model_path_template.format(
        model_name=config.model_name,
        image_size=config.image_size,
        f1_score=f1_score
    )
    torch.save(model_state, final_model_path)
    logger.info(f"Final model saved as {final_model_path}")

def load_model(model_path, model, optimizer=None):
    """
    Loads a model and its optimizer state from a checkpoint file.

    Parameters:
        model_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model instance to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance to load the state into, if provided.

    Returns:
        best_f1_score (float): The best F1 score recorded in the checkpoint.
        epochs (int): The number of epochs the model was trained for.
    """
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_f1_score = checkpoint.get('f1_score', -1)  # Default to -1 if not present
    epochs = checkpoint.get('epoch', 0)  # Default to 0 if not present
    
    return best_f1_score, epochs