from torchvision import models as models
import torch.nn as nn
import torch
import timm
from imclaslib.models.ensemble_classifier import EnsembleClassifier
from imclaslib.models.gcn_classifier import GCNClassifier
from imclaslib.models.multilabel_classifier import MultiLabelClassifier
from imclaslib.models.multilabel_embeddinglayer_model import MultiLabelClassifier_LabelEmbeddings

def create_model(config):
    """
    Creates a PyTorch multi-label image classification model with optional label embedding.

    Parameters:
        config (config) The config used to load create the model

    Returns:
        nn.Module: The PyTorch model with the configured classifier head.
    """
    if config.model_ensemble_model_configs:
        return EnsembleClassifier(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_features = None
    # Try to load the model from torchvision.models
    try:
        # Use the 'models' module from torchvision
        model = getattr(models, config.model_name)(weights=config.model_weights)
    except AttributeError:
        # If the model is not available in torchvision, try loading it from timm
        if timm.is_model(config.model_name):
            # Use timm to create the model without the classifier (head)
            model = timm.create_model(
                config.model_name,
                pretrained=True,
                num_classes=0  # Setting num_classes=0 removes the classifier
            )
            num_features = model.num_features  # Get the number of features after pooling
        else:
            raise ValueError(f"The model '{config.model_name}' is not available in torchvision or timm.")
    model = model.to(device)

    # Freeze or unfreeze the model parameters based on requires_grad
    for param in model.parameters():
        param.requires_grad = config.train_requires_grad

    # Replace the appropriate classifier head with a new one
    if num_features is None:
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            num_features = model.classifier[-1].in_features
            #print(f"Number of input features to the classifier: {num_features}")
            model.classifier[-1] = nn.Identity()
        elif hasattr(model, 'fc'):
            num_features = model.fc.in_features
            #print(f"Number of input features to the fc: {num_features}")
            model.fc = nn.Identity()
        elif hasattr(model, 'head'):
            num_features = model.head.in_features
            #print(f"Number of input features to the head: {num_features}")
            model.head = nn.Identity()
        elif hasattr(model, 'heads') and isinstance(model.heads, nn.Sequential):
            num_features = model.heads[-1].in_features
            #print(f"Number of input features to the heads: {num_features}")
            model.heads[-1] = nn.Identity()
        else:
            raise AttributeError(f"The model '{config.model_name}' does not have a recognized classifier head.")
    model.output_dim = num_features 

    # If add_embedding_layer is True, wrap the base model with the MultiLabelClassifier_LabelEmbeddings
    if config.model_embedding_layer_enabled:
        model = MultiLabelClassifier_LabelEmbeddings(model, config.model_num_classes, config.model_embedding_layer_dimension, config.train_dropout_prob)
    elif config.model_gcn_enabled:
        if config.model_gcn_model_name is None:
            raise ValueError("GCN model name must be provided when use_gcn is True.")
        gcn_model_params = {
            'in_channels': config.model_embedding_layer_dimension,
            'out_channels': config.model_gcn_out_channels,  # Output dimension size (should match base model output dimension for concatenation)
            'dropout': config.train_dropout_prob / 100,
            'hidden_channels': config.model_embedding_layer_dimension,
            'num_layers': config.model_gcn_layers
        }
        config.model_gcn_edge_index = config.model_gcn_edge_index.to(device)
        if config.model_gcn_edge_weights is not None:
            config.model_gcn_edge_weights = config.model_gcn_edge_weights.to(device)
        model = GCNClassifier(
            base_model=model, 
            num_classes=config.model_num_classes, 
            model_gcn_model_name=config.model_gcn_model_name, 
            dropout_prob=config.train_dropout_prob / 100, 
            gcn_model_params=gcn_model_params,
            edge_index=config.model_gcn_edge_index,
            edge_weight=config.model_gcn_edge_weights,
            num_heads=config.model_attention_layer_num_heads
        )
    else:
        # If not using the embedding layer, create a MultiLabelClassifier with dropout
        model = MultiLabelClassifier(model, config.model_num_classes, config.train_dropout_prob / 100)
        #print(model)
    return model