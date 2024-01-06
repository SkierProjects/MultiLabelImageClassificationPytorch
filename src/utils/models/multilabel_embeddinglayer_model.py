import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        assert self.head_dim * num_heads == self.embedding_dim, "embedding_dim must be divisible by num_heads"

        self.query = nn.Linear(feature_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, features, label_embeddings):
        batch_size = features.shape[0]

        # Linear projections
        query = self.query(features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(label_embeddings).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        value = self.value(label_embeddings).view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        # Attention scores and softmax
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_distribution = F.softmax(attention_scores, dim=-1)

        # Concatenate heads and put through final linear layer
        attention_output = torch.matmul(attention_distribution, value).transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.embedding_dim)
        output = self.out(attention_output)

        return output

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
        self.attention = MultiHeadAttention(embedding_dim, embedding_dim, 4)

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
            #taking mean here is likely not ideal.
            #label_embeddings = self.label_embedding.weight.mean(dim=0, keepdim=True)  # [1, embedding_dim]
            #label_embeddings = label_embeddings.expand(transformed_image_features.size(0), -1)  # [batch_size, embedding_dim]

            attention_weights = self.attention(transformed_image_features, self.label_embedding.weight)  # [batch_size, num_labels]
            label_embeddings = torch.matmul(attention_weights, self.label_embedding.weight)  # [batch_size, embedding_dim]

        # Combine the image features with the label embeddings
        combined_features = transformed_image_features + label_embeddings
        
        # Pass the combined features through the classifier
        logits = self.classifier(combined_features)  # [batch_size, num_classes]
        return logits