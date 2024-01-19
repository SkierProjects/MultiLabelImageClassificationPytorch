import torch
from imclaslib.logging.loggerfactory import LoggerFactory
import imclaslib.files.pathutils as pathutils
import os
import re

logger = LoggerFactory.get_logger(f"logger.{__name__}")

def save_best_model(model_state, config):
    """
    Saves the best model state to the predetermined best model path.

    Parameters:
        model_state (dict): State dictionary of the model to be saved.
    """
    torch.save(model_state, pathutils.get_best_model_path(config))

def save_final_model(model_state, f1_score, config):
    """
    Saves the final model state to a filename that includes model details such as name, image size, and F1 score.

    Parameters:
        model_state (dict): State dictionary of the model to be saved.
        f1_score (float): The F1 score of the model.
        config (object): Configuration object containing model_name and image_size.
    """
    modelAddons = ""
    if config.model_embedding_layer_enabled:
        modelAddons = "_EmbeddingLayer"
    elif config.model_gcn_enabled:
        modelAddons = "_GCN"
    final_model_path_template = os.path.join(str(pathutils.get_output_dir_path(config)), '{model_name}_{image_size}_{f1_score:.4f}{modelAddons}.pth')
    final_model_path = final_model_path_template.format(
        model_name=config.model_name,
        image_size=config.model_image_size,
        f1_score=f1_score,
        modelAddons=modelAddons
    )
    torch.save(model_state, final_model_path)
    logger.info(f"Final model saved as {final_model_path}")

def load_model(model_path, config):
    """
    Loads a model and its optimizer state from a checkpoint file.

    Parameters:
        model_path (str): Path to the checkpoint file.
        config (object): Configuration object.

    Returns:
        model_data (dict): The model data from the file.
    """
    checkpoint = torch.load(model_path)
    
    model_data = add_model_data(checkpoint, config)
    return model_data

def add_model_data(checkpoint, config):
    model_data = {}
    model_data["epoch"] = checkpoint.get('epoch', 0)
    model_data["model_state_dict"] = checkpoint.get('model_state_dict', -1)
    model_data["optimizer_state_dict"] = checkpoint.get('optimizer_state_dict', -1)
    model_data["loss"] = checkpoint.get('loss', -1)
    model_data["f1_score"] = checkpoint.get('f1_score', -1)
    model_data["model_name"] = checkpoint.get('model_name', config.model_name)
    model_data["requires_grad"] = checkpoint.get('requires_grad', True)
    model_data["model_num_classes"] = checkpoint.get('model_num_classes', config.model_num_classes)
    model_data["dropout"] = checkpoint.get('dropout', 0)
    model_data["embedding_layer"] = checkpoint.get('embedding_layer', config.model_embedding_layer_enabled)
    model_data["model_gcn_enabled"] = checkpoint.get('model_gcn_enabled', config.model_gcn_enabled)
    model_data["train_batch_size"] = checkpoint.get('train_batch_size', config.train_batch_size)
    model_data["optimizer"] = checkpoint.get('optimizer', 'Adam')
    model_data["loss_function"] = checkpoint.get('loss_function', 'BCEWithLogitsLoss')
    model_data["image_size"] = checkpoint.get('image_size', config.model_image_size)
    model_data["model_gcn_model_name"] = checkpoint.get('model_gcn_model_name', config.model_gcn_model_name)
    model_data["model_gcn_out_channels"] = checkpoint.get('model_gcn_out_channels', config.model_gcn_out_channels)
    model_data["model_gcn_layers"] = checkpoint.get('model_gcn_layers', config.model_gcn_layers)
    model_data["model_attention_layer_num_heads"] = checkpoint.get('model_attention_layer_num_heads', config.model_attention_layer_num_heads)
    model_data["model_embedding_layer_dimension"] = checkpoint.get('model_embedding_layer_dimension', config.model_embedding_layer_dimension)
    model_data["train_loss"] = checkpoint.get('train_loss', 0)

    
    return model_data

def update_config_from_model_file(config):
    pattern = r"(.+?)_(\d{3})_\d\.\d{4}"
    file_name = config.model_name_to_load
    if not file_name:
        return
    match = re.match(pattern, file_name)
    if match:
        # If there's a match, get the model name and image size
        model_name = match.group(1)  # The first capture group (modelname)
        model_image_size = match.group(2)  # The second capture group (image size)
        config.model_name = model_name
        config.model_image_size = int(model_image_size)
        return
    else:
        model_file_path = pathutils.get_model_to_load_path(config)
        checkpoint = torch.load(model_file_path)
        model_name = checkpoint.get('model_name', None)
        model_image_size = checkpoint.get('image_size', None)
        if model_name is not None:
            config.model_name = model_name
        if model_image_size is not None:
            config.model_image_size = model_image_size
        return
    
def load_pretrained_weights_exclude_classifier(new_model, config, freeze_base_model=False):
    pretrained_model_path = pathutils.combine_path(pathutils.get_output_dir_path(config), f"{config.train_model_to_load_raw_weights}.pth")
    path = str(pretrained_model_path)
    # Load the state dictionary of the pretrained model
    pretrained_state_dict = torch.load(path)
    model_data = add_model_data(pretrained_state_dict, config)
    # Remove the weights for the final classifier layer from the pretrained state_dict
    classifier_keys = [key for key in pretrained_state_dict if key.startswith('classifier.') or key.startswith('fc.') or key.startswith('head.') or key.startswith('heads.') ]
    for key in classifier_keys:
        pretrained_state_dict.pop(key)

    # Load the remaining weights into the new model's base model
    # This will exclude the final classifier layer
    new_model.base_model.load_state_dict(pretrained_state_dict, strict=False)

    # Freeze the parameters of the base model, if required
    if freeze_base_model:
        for param in new_model.base_model.parameters():
            param.requires_grad = False

    return new_model, model_data