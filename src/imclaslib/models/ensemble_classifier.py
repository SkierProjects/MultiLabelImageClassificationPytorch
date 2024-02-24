import torch.nn as nn
import torch
from imclaslib.files import pathutils
from imclaslib.files import modelloadingutils
from imclaslib.metrics import metricutils
import imclaslib.models.modelfactory as modelfactory

class EnsembleClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        mode = config.model_ensemble_combiner
        self.mode = mode
        # We need to use ModuleList so that the models are properly registered as submodules of the ensemble
        self.models = nn.ModuleList()
        self.temperatures = []
        for modelconfig in config.model_ensemble_model_configs:
            modelloadingutils.update_config_from_model_file(modelconfig)
            model = modelfactory.create_model(modelconfig)
            modelToLoadPath = pathutils.get_model_to_load_path(modelconfig)
            modelData = modelloadingutils.load_model(modelToLoadPath, modelconfig)
            model.load_state_dict(modelData['model_state_dict'])
            for param in model.parameters():
                param.requires_grad = False  # Freeze the model parameters
            self.models.append(model)
            self.temperatures.append(modelconfig.model_temperature)
        
        num_models = len(config.model_ensemble_model_configs)
        num_classes = config.model_num_classes

        if mode == 'einsum':
            self.meta_weights = nn.Parameter(torch.ones(num_models, num_classes))
        elif mode == 'linear':
            self.combining_layer = nn.Linear(num_models * num_classes, num_classes)

    def forward(self, x):
        logits_list = []
        for i, model in enumerate(self.models):
            model_logits = model(x)
            if self.temperatures[i] != None:
                model_logits = metricutils.temperature_scale(model_logits, self.temperatures[i])
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