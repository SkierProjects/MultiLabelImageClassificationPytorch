import torch.nn as nn
import torch

class MultiLabelClassifier(nn.Module):
    def __init__(self, base_model, num_classes, dropout_prob):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Assuming base_model outputs features of size (batch_size, feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(base_model.output_dim, num_classes)
        )
        self.output_dim = num_classes  # Set the output dimension

    def forward(self, x):
        # Get the image features from the base model
        image_features = self.base_model(x)  # [batch_size, feature_dim]
        
        # Pass the image features through the classifier
        logits = self.classifier(image_features)  # [batch_size, num_classes]
        
        return logits