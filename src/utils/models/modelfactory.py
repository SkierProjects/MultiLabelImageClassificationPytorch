from torchvision import models as models
import torch.nn as nn

def create_model(model_name, requires_grad, num_classes, dropout_prob=0.0, weights=None):
    """
    Creates a PyTorch model with the specified configuration.

    Parameters:
        model_name (str): Name of the model to create.
        requires_grad (bool): Whether the model parameters should be trainable.
        num_classes (int): Number of classes for the final classification layer.
        dropout_prob (float): Dropout probability for the new classifier head. Default is 0.0.
        weights (PretrainedWeights or bool): Pretrained weights to initialize the model. Default is None.

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
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'fc'):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'head'):
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'heads') and isinstance(model.heads, nn.Sequential):
        # Access the head and replace it with a new Sequential module with dropout and linear layers
        num_features = model.heads[0].in_features
        model.heads = nn.Sequential(
            nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity(),
            nn.Linear(num_features, num_classes)
        )
    else:
        raise AttributeError(f"The model '{model_name}' does not have a recognized classifier head.")

    return model