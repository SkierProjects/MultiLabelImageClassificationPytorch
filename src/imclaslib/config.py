import json
import torch
import yaml
import copy

class Config:
    def __init__(self, default_config_path=None):
        """
        Initialize a new Config instance, optionally loading values from a default JSON/YAML file.

        Parameters:
        - default_config_path: str (optional), path to a JSON/YAML file with default configuration values.
        """

        # model - needed to define a model
        self.model_name = 'efficientnet_v2_m'
        self.model_image_size = 400
        self.model_num_classes = 31
        self.model_weights = 'DEFAULT'
        self.model_folder = ""
        self.model_tags_path = ""       
        self.model_name_to_load = "best_model"
        self.model_attention_layer_num_heads = 8
        self.model_ensemble_combiner = "mean"
        self.model_ensemble_model_configs = None
        self.model_fp16 = False

        #model - embedding layer
        self.model_embedding_layer_enabled = False
        self.model_embedding_layer_dimension = 512

        #model - gcn
        self.model_gcn_enabled = False
        self.model_gcn_model_name = "GAT"
        self.model_gcn_out_channels = 512
        self.model_gcn_layers = 4
        self.model_gcn_graph_path = ""
        self.model_gcn_edge_index = None
        self.model_gcn_edge_weights = None

        # dataset
        self.dataset_path = ""
        self.dataset_augmentation_level = 0
        self.dataset_normalization_mean = [0.4805, 0.3967, 0.3589]
        self.dataset_normalization_std = [0.3207, 0.2930, 0.2824]
        self.dataset_train_percentage = 80
        self.dataset_valid_percentage = 10
        self.dataset_test_percentage = 10
        self.dataset_version = 2.0
        self.dataset_tags_mapping_dict = {}

        # training
        self.train_batch_size = 24
        self.train_dropout_prob = 0
        self.train_learning_rate = 1e-4
        self.train_num_epochs = 50
        self.train_continue_training = False
        self.train_requires_grad = True
        self.train_store_gradients_epoch_interval = 5
        self.train_check_test_loss_epoch_interval = 10
        self.train_many_models_path = ""
        self.train_model_to_load_raw_weights = ""
        self.train_l2_enabled = False
        self.train_l2_lambda = 0.01

        # training - early stopping
        self.train_early_stopping_patience = 6
        self.train_early_stopping_threshold = 4e-3

        #training - learning rate reducer
        self.train_learningrate_reducer_patience = 2
        self.train_learningrate_reducer_threshold = 2e-3
        self.train_learningrate_reducer_factor = 0.1
        self.train_learningrate_reducer_min_lr = 1e-7

        #test
        self.test_batch_size = 72
        self.test_many_models_path = ""

        #logs
        self.logs_level = "DEBUG"
        self.logs_folder = ""
        self.logs_tensorboard_folder = ""
        self.project_name = ""

        if default_config_path:
            self.load_config(default_config_path)


    def load_config(self, config_path):
        """
        Load configuration data from a file (JSON or YAML) based on its extension.

        Parameters:
        - config_path: str, path to the JSON/YAML file with configuration values.
        """
        extension = config_path.split('.')[-1].lower()
        if extension == 'json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif extension in ['yaml', 'yml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {extension}")

        self.update_config(config_data, self)

    def update_config(self, new_config, default_config):
        self.__update_config(new_config)
        if self.model_gcn_edge_index is not None and not isinstance(self.model_gcn_edge_index, torch.Tensor):
            self.model_gcn_edge_index = torch.tensor(self.model_gcn_edge_index)

        if self.model_gcn_edge_weights is not None and not isinstance(self.model_gcn_edge_weights, torch.Tensor):
            self.model_gcn_edge_weights = torch.tensor(self.model_gcn_edge_weights)

        if self.model_ensemble_model_configs is not None:
            self.model_ensemble_model_configs = [Config.from_dict(ensemble_config_data, default_config) for ensemble_config_data in self.model_ensemble_model_configs]

    def __update_config(self, new_config, prefix=''):
        """
        Update the configuration instance with new values.

        Parameters:
        - new_config: dict, new configuration values to update with.
        - prefix: str, prefix for nested attributes to maintain hierarchy.
        """
        if new_config:
            for key, value in new_config.items():
                if isinstance(value, dict):  # It's a subsection
                    new_prefix = f"{prefix}{key}_"
                    self.__update_config(value, new_prefix)
                else:
                    config_key = f"{prefix}{key}"
                    if hasattr(self, config_key):
                        setattr(self, config_key, value)

    def __getattr__(self, name):
        """
        Allow dynamic access to configuration values.
        """
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    @classmethod
    def from_dict(cls, config_dict, default_config=None):
        if default_config is None:
            new_instance = cls()
        else:
            new_instance = copy.deepcopy(default_config)

        new_instance.update_config(config_dict, default_config)
        return new_instance

    @staticmethod
    def load_configs_from_file(file_path, default_config):
        """
        Load model configurations from a JSON or yaml file.
        
        Parameters:
        - file_path: str, the path to the JSON or yaml file containing the configurations
        
        Returns:
        - list, the list of configuration objects
        """
        extension = str(file_path).split('.')[-1].lower()
        if extension == 'json':
            with open(file_path, 'r') as f:
                config_data = json.load(f)
        elif extension in ['yaml', 'yml']:
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {extension}")
        return [Config.from_dict(config, default_config) for config in config_data]