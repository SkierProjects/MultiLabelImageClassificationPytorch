import torch.nn as nn
import torch

class MultiLabelClassifier_LabelEmbeddings(nn.Module):
    def __init__(self, base_model, num_classes, embedding_dim, dropout_prob):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Embedding layer for labels
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        # Assuming base_model outputs features of size (batch_size, feature_dim)
        self.feature_transform = nn.Linear(base_model.output_dim, embedding_dim)

        # Classifier head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Dropout layer (if dropout_prob is greater than 0)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()

    def forward(self, x):
        # Get the image features from the base model
        image_features = self.base_model(x)

        # Transform image features to the same dimension as label embeddings
        transformed_features = self.feature_transform(image_features)
        transformed_features = self.dropout(transformed_features)

        # Calculate the dot product between transformed features and label embeddings
        # This step combines image features with the label embedding space
        label_embeddings = self.label_embedding.weight
        combined_features = torch.matmul(transformed_features, label_embeddings.t())

        # Classifier to produce final outputs
        logits = self.classifier(combined_features)

        # Apply sigmoid activation for multi-label classification
        return torch.sigmoid(logits)
