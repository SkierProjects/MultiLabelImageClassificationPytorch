import torch
import torch.nn as nn
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
        key = self.key(label_embeddings).view(self.num_heads, -1, self.head_dim)
        value = self.value(label_embeddings).view(self.num_heads, -1, self.head_dim)

        # Attention scores and softmax
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_distribution = F.softmax(attention_scores, dim=-1)

        # Concatenate heads and put through final linear layer
        attention_output = torch.matmul(attention_distribution, value).transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        output = self.out(attention_output)

        return output.squeeze(1)  # Ensure output is [batch_size, embedding_dim]
    
class Attention(nn.Module):
    def __init__(self, image_features_dim, label_embedding_dim):
        super().__init__()
        self.image_to_query = nn.Linear(image_features_dim, label_embedding_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, label_embeddings):
        query = self.image_to_query(features).unsqueeze(1)  # [batch_size, 1, label_embedding_dim]
        # Use label_embeddings as key and value
        key_value = label_embeddings  # [num_classes, label_embedding_dim]

        # Compute attention scores
        attention_scores = torch.matmul(query, key_value.transpose(0, 1))  # [batch_size, 1, num_classes]

        # Apply softmax to get probabilities
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to key_value
        context_vector = torch.matmul(attention_weights, key_value).squeeze(1)  # [batch_size, label_embedding_dim]

        return context_vector