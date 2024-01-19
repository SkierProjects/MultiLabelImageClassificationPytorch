import torch_geometric.nn as GCN
import torch.nn as nn
import torch

from imclaslib.models.model_layers import Attention, MultiHeadAttention

class GCNClassifier(nn.Module):
    def __init__(self, base_model, num_classes, gcn_model_name, dropout_prob, gcn_model_params, edge_index, edge_weight=None, use_multihead_attention=True, num_heads=4):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        
        # Instantiate the pre-made GCN model from PyTorch Geometric
        self.gcn = getattr(GCN, gcn_model_name)(**gcn_model_params)

        # Store the graph structure
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Final classifier layer
        self.classifier = nn.Linear(base_model.output_dim + gcn_model_params['out_channels'], num_classes)

        # Initialize a placeholder for label embeddings, which will be learned during training
        self.label_embeddings = nn.Parameter(torch.Tensor(num_classes, gcn_model_params['in_channels']))
        nn.init.xavier_uniform_(self.label_embeddings)

        # Initialize the appropriate attention mechanism
        if use_multihead_attention:
            self.attention = MultiHeadAttention(base_model.output_dim, gcn_model_params['out_channels'], num_heads)
        else:
            self.attention = Attention(base_model.output_dim, gcn_model_params['out_channels'])

    def forward(self, x, labels=None):
        # Get the image features from the base model
        image_features = self.base_model(x)  # [batch_size, feature_dim]

        # Update label_embeddings using the GCN and the graph structure
        label_embeddings_updated = self.gcn(self.label_embeddings, self.edge_index, self.edge_weight)

        if self.training and labels is not None:
            # Use the provided labels to select the relevant embeddings for each example in the batch
            batch_label_embeddings = torch.matmul(labels.float(), label_embeddings_updated)
        else:
            # During inference or if labels are not provided, use the attention mechanism
            batch_label_embeddings = self.attention(image_features, label_embeddings_updated)

        # Combine the image features with the label embeddings
        combined_features = torch.cat((image_features, batch_label_embeddings), dim=1)
        combined_features = self.dropout(combined_features)
        
        # Pass the combined features through the classifier
        logits = self.classifier(combined_features)  # [batch_size, num_classes]
        return logits