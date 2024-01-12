from torchvision import models as models
import torch.nn as nn
import torch
from utils.models.ensemble_classifier import EnsembleClassifier
from utils.models.gcn_classifier import GCNClassifier
from utils.models.multilabel_classifier import MultiLabelClassifier
from utils.models.multilabel_embeddinglayer_model import MultiLabelClassifier_LabelEmbeddings

def create_model(config):
    """
    Creates a PyTorch multi-label image classification model with optional label embedding.

    Parameters:
        model_name (str): Name of the model to create.
        requires_grad (bool): Whether the model parameters should be trainable.
        num_classes (int): Number of classes for the final classification layer.
        dropout_prob (float): Dropout probability for the new classifier head. Default is 0.0.
        weights (PretrainedWeights or bool): Pretrained weights to initialize the model. Default is None.
        add_embedding_layer (bool): Whether to add an embedding layer for label context. Default is False.
        embedding_dim (int): Dimension of the label embedding. Default is 128.

    Returns:
        nn.Module: The PyTorch model with the configured classifier head.
    """
    if config.ensemble_model_configs:
        return EnsembleClassifier(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the specified model with the specified pretrained weights if required
    model = getattr(models, config.model_name)(weights=config.model_weights)
    model = model.to(device)

    # Freeze or unfreeze the model parameters based on requires_grad
    for param in model.parameters():
        param.requires_grad = config.model_requires_grad

    # Replace the appropriate classifier head with a new one
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    elif hasattr(model, 'fc'):
        num_features = model.fc.in_features
        model.fc = nn.Identity()
    elif hasattr(model, 'head'):
        num_features = model.head.in_features
        model.head = nn.Identity()
    elif hasattr(model, 'heads') and isinstance(model.heads, nn.Sequential):
        num_features = model.heads[0].in_features
        model.heads = nn.Identity()
    else:
        raise AttributeError(f"The model '{config.model_name}' does not have a recognized classifier head.")
    model.output_dim = num_features 

    # If add_embedding_layer is True, wrap the base model with the MultiLabelClassifier_LabelEmbeddings
    if config.embedding_layer_enabled:
        model = MultiLabelClassifier_LabelEmbeddings(model, config.num_classes, config.embedding_layer_dimension, config.model_dropout_prob)
    elif config.gcn_enabled:
        if config.gcn_model_name is None:
            raise ValueError("GCN model name must be provided when use_gcn is True.")
        gcn_model_params = {
            'in_channels': config.embedding_layer_dimension,
            'out_channels': config.gcn_out_channels,  # Output dimension size (should match base model output dimension for concatenation)
            'dropout': config.model_dropout_prob / 100,
            'hidden_channels': config.embedding_layer_dimension,
            'num_layers': config.gcn_layers
        }
        config.gcn_edge_index = config.gcn_edge_index.to(device)
        if config.gcn_edge_weights is not None:
            config.gcn_edge_weights = config.gcn_edge_weights.to(device)
        model = GCNClassifier(
            base_model=model, 
            num_classes=config.num_classes, 
            gcn_model_name=config.gcn_model_name, 
            dropout_prob=config.model_dropout_prob / 100, 
            gcn_model_params=gcn_model_params,
            edge_index=config.gcn_edge_index,
            edge_weight=config.gcn_edge_weights,
            num_heads=config.attention_layer_num_heads
        )
    else:
        # If not using the embedding layer, create a MultiLabelClassifier with dropout
        model = MultiLabelClassifier(model, config.num_classes, config.model_dropout_prob / 100)

    return model