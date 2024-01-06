from torchvision import models as models
import torch.nn as nn
from utils.models.gcn_classifier import GCNClassifier
from utils.models.multilabel_classifier import MultiLabelClassifier
from utils.models.multilabel_embeddinglayer_model import MultiLabelClassifier_LabelEmbeddings

def create_model(model_name, requires_grad, num_classes, dropout_prob=0.0, weights=None, add_embedding_layer=False, embedding_dim=128, use_gcn=False, gcn_model_name="GCN"):
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
    # Load the specified model with the specified pretrained weights if required
    model = getattr(models, model_name)(weights=weights)

    # Freeze or unfreeze the model parameters based on requires_grad
    for param in model.parameters():
        param.requires_grad = requires_grad

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
        raise AttributeError(f"The model '{model_name}' does not have a recognized classifier head.")
    model.output_dim = num_features 

    # If add_embedding_layer is True, wrap the base model with the MultiLabelClassifier_LabelEmbeddings
    if add_embedding_layer:
        model = MultiLabelClassifier_LabelEmbeddings(model, num_classes, embedding_dim, dropout_prob)
    elif use_gcn:
        if gcn_model_name is None:
            raise ValueError("GCN model name must be provided when use_gcn is True.")
        model = GCNClassifier(model, num_classes, gcn_model_name, dropout_prob)
    else:
        # If not using the embedding layer, create a MultiLabelClassifier with dropout
        model = MultiLabelClassifier(model, num_classes, dropout_prob)

    return model