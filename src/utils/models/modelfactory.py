from torchvision import models as models
import torch.nn as nn

from utils.models.multilabel_embeddinglayer_model import MultiLabelClassifier_LabelEmbeddings

def create_model(model_name, requires_grad, num_classes, dropout_prob=0.0, weights=None, add_embedding_layer=False, embedding_dim=62):
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
    base_model = getattr(models, model_name)(weights=weights)
    
    # Freeze or unfreeze the model parameters based on requires_grad
    for param in base_model.parameters():
        param.requires_grad = requires_grad

    # Replace the model's classifier head with the new classifier
    if hasattr(base_model, 'classifier') and isinstance(base_model.classifier, nn.Sequential):
        base_model.output_dim = base_model.classifier[-1].in_features
        classifier = nn.Linear(base_model.output_dim, num_classes)
    elif hasattr(base_model, 'fc'):
        base_model.output_dim = base_model.fc.in_features
        classifier = nn.Linear(base_model.output_dim, num_classes)
    elif hasattr(base_model, 'head'):
        base_model.output_dim = base_model.head.in_features
        classifier = nn.Linear(base_model.output_dim, num_classes)
    else:
        raise AttributeError(f"The model '{model_name}' does not have a recognized classifier head.")

    if add_embedding_layer:
        # Use the MultiLabelClassifier with label embeddings
        model = MultiLabelClassifier_LabelEmbeddings(base_model, num_classes, embedding_dim, dropout_prob)
    else:
        # Use a simple classifier head
        if dropout_prob > 0.0:
            classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                classifier
            )
        # Attach the classifier to the base model
        if hasattr(base_model, 'classifier'):
            base_model.classifier = classifier
        elif hasattr(base_model, 'fc'):
            base_model.fc = classifier
        elif hasattr(base_model, 'head'):
            base_model.head = classifier

        model = base_model

    return model