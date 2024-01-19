import torch.nn as nn
import torch
from imclaslib.models.model_layers import MultiHeadAttention, Attention

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

        # Batch normalization layer after feature transformation
        #self.batch_norm = nn.BatchNorm1d(embedding_dim)

        # Classifier head, which maps the concatenated embeddings to the output space
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()

        # Attention layer
        self.attention = Attention(embedding_dim, embedding_dim)

    def forward(self, x, labels=None):
        # Get the image features from the base model
        image_features = self.base_model(x)  # [batch_size, feature_dim]

        # Transform image features to the same dimension as label embeddings
        transformed_image_features = self.feature_transform(image_features)  # [batch_size, embedding_dim]

        # Apply batch normalization
        #transformed_image_features = self.batch_norm(transformed_image_features)
        transformed_image_features = self.dropout(transformed_image_features)

        if labels is not None:
            # During training, use the one-hot encoded labels to compute the label embeddings
            label_embeddings = torch.matmul(labels, self.label_embedding.weight)  # [batch_size, embedding_dim]
        else:
            # We don't need to unsqueeze and squeeze since Attention now expects 2D tensors
            attention_output = self.attention(transformed_image_features, self.label_embedding.weight)
            label_embeddings = attention_output  # This is now [batch_size, embedding_dim]

        # Combine the image features with the label embeddings
        combined_features = transformed_image_features + label_embeddings
        
        # Pass the combined features through the classifier
        logits = self.classifier(combined_features)  # [batch_size, num_classes]
        return logits