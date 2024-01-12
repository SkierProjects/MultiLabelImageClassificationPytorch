import torch_geometric.nn as GCN
import torch.nn as nn
import torch
from utils.files import pathutils
from utils.files import modelloadingutils
import utils.models.modelfactory as modelfactory

from utils.models.model_layers import Attention, MultiHeadAttention

class EnsembleClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        # We need to use ModuleList so that the models are properly registered as submodules of the ensemble
        self.models = nn.ModuleList()
        for modelconfig in config.ensemble_model_configs:
            modelloadingutils.update_config_from_model_file(modelconfig)
            model = modelfactory.create_model(modelconfig)
            modelToLoadPath = pathutils.get_model_to_load_path(modelconfig)
            modelData = modelloadingutils.load_model(modelToLoadPath, modelconfig)
            model.load_state_dict(modelData['model_state_dict'])
            for param in model.parameters():
                param.requires_grad = False  # Freeze the model parameters
            self.models.append(model)
        num_models = len(config.ensemble_model_configs)
        num_classes = config.num_classes
        self.meta_weights = nn.Parameter(torch.ones(num_models, num_classes))
        #self.combining_layer = nn.Linear(num_models * num_classes, num_classes)

    def forward(self, x):
        # Initialize a list to hold the logits from each model
        logits_list = []
        # Iterate over the models, pass the input through each, and collect the logits
        for model in self.models:
            # We assume each model returns logits directly; if not, adjust accordingly
            model_logits = model(x)
            logits_list.append(model_logits)

        # Stack the logits along a new dimension to create a tensor of shape [num_models, batch_size, num_classes]
        stacked_logits = torch.stack(logits_list, dim=0)
        
        # Apply meta-learner weights to the stacked logits
        weighted_logits = torch.einsum('mnc,mc->mnc', stacked_logits, self.meta_weights)
        logits = torch.mean(weighted_logits, dim=0)
            

        # Concatenate the logits along the last dimension to create a single flat vector for each example
        #concatenated_logits = torch.cat(logits_list, dim=-1)
        
        # Pass the concatenated logits through the linear layer to obtain final predictions
        #logits = self.combining_layer(concatenated_logits)
        return logits