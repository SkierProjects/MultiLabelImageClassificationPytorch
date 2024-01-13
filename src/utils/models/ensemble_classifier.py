import torch_geometric.nn as GCN
import torch.nn as nn
import torch
from utils.files import pathutils
from utils.files import modelloadingutils
import utils.models.modelfactory as modelfactory

from utils.models.model_layers import Attention, MultiHeadAttention

class EnsembleClassifier(nn.Module):
    def __init__(self, config, mode='linear'):
        super().__init__()
        self.mode = mode
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

        if mode == 'einsum':
            self.meta_weights = nn.Parameter(torch.ones(num_models, num_classes))
        elif mode == 'linear':
            self.combining_layer = nn.Linear(num_models * num_classes, num_classes)

    def forward(self, x):
        logits_list = []
        for model in self.models:
            model_logits = model(x)
            logits_list.append(model_logits)

        if self.mode == 'einsum':
            stacked_logits = torch.stack(logits_list, dim=0)
            weighted_logits = torch.einsum('mnc,mc->mnc', stacked_logits, self.meta_weights)
            logits = torch.mean(weighted_logits, dim=0)
        elif self.mode == 'linear':
            concatenated_logits = torch.cat(logits_list, dim=-1)
            logits = self.combining_layer(concatenated_logits)
        elif self.mode == 'mean':
            stacked_logits = torch.stack(logits_list, dim=0)
            logits = torch.mean(stacked_logits, dim=0)
        elif self.mode == 'max':
            stacked_logits = torch.stack(logits_list, dim=0)
            logits, _ = torch.max(stacked_logits, dim=0)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return logits