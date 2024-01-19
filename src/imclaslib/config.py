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
        self.model_name = 'efficientnet_v2_m'
        self.model_requires_grad = True
        self.num_classes = 31
        self.model_dropout_prob = 0
        self.model_weights = 'DEFAULT'
        self.model_image_size = 400
        self.batch_size = 24
        self.learning_rate = 1e-4
        self.num_epochs = 50
        self.continue_training = False
        self.model_name_to_load = "best_model"
        self.early_stopping_patience = 6
        self.early_stopping_threshold = 4e-3
        self.learningrate_reducer_patience = 2
        self.learningrate_reducer_threshold = 2e-3
        self.learningrate_reducer_factor = 0.1
        self.learningrate_reducer_min_lr = 1e-7
        self.augmentation_level = 0
        self.embedding_layer_enabled = False
        self.embedding_layer_dimension = 512
        self.gcn_enabled = False
        self.gcn_model_name = "GAT"
        self.gcn_out_channels = 512
        self.gcn_layers = 4
        self.attention_layer_num_heads = 8
        # Define the edges
        self.gcn_edge_index = torch.tensor(
            [[18, 19, 13, 19, 26, 19,  2,  3, 14, 19,  1, 19, 29, 19,  8, 19, 17, 26,
                21, 19,  6, 19,  2, 19,  2,  1, 15, 19,  5, 19,  7, 19,  2, 21, 25, 19,
                25,  8, 16, 19, 10, 14,  8,  6, 15,  8, 14,  8, 22, 23,  6, 14,  8, 13,
                16,  0,  8, 16,  8,  5,  7, 18,  5, 15, 23, 28,  2,  0, 11, 19, 14,  1,
                5, 16, 14, 29, 17, 19, 23, 11,  6,  7, 22,  0, 22, 19, 30, 19, 28, 19,
                22, 30,  5,  0, 21,  0, 28, 28, 28, 10, 19, 16,  9,  5, 24, 21, 24, 30,
                28, 21, 18,  0, 19, 11,  3,  3,  3,  5],
            [19, 18, 19, 13, 19, 26,  3,  2, 19, 14, 19,  1, 19, 29, 19,  8, 26, 17,
                19, 21, 19,  6, 19,  2,  1,  2, 19, 15, 19,  5, 19,  7, 21,  2, 19, 25,
                8, 25, 19, 16, 14, 10,  6,  8,  8, 15,  8, 14, 23, 22, 14,  6, 13,  8,
                0, 16, 16,  8,  5,  8, 18,  7, 15,  5, 28, 23,  0,  2, 19, 11,  1, 14,
                16,  5, 29, 14, 19, 17, 11, 23,  7,  6,  0, 22, 19, 22, 19, 30, 19, 28,
                30, 22,  0,  5,  0, 21, 11, 10, 29, 19, 10,  9, 16, 24,  5, 24, 21, 28,
                30, 18, 21, 19,  0, 12,  1, 11, 10,  9]])
        self.gcn_edge_weights = torch.tensor(
            [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
            1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
            1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
            1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
            1.0000,  1.0000,  1.0000,  1.0000,  0.5000,  0.5000,  0.5000,  0.5000,
            0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,
            0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,
            0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,
            0.5000,  0.5000,  0.5000,  0.5000,  0.8000,  0.8000,  0.8000,  0.8000,
            0.8000,  0.8000,  0.8000,  0.8000,  0.8000,  0.8000,  0.8000,  0.8000,
            0.8000,  0.8000,  0.8000,  0.8000,  0.8000,  0.8000,  0.8000,  0.8000,
            0.8000,  0.8000,  0.8000,  0.8000,  0.8000,  0.8000,  0.8000,  0.8000,
            1.0000,  1.0000,  1.0000, -0.5000, -0.5000, -0.5000, -0.5000, -0.5000,
            -0.5000, -0.5000, -0.5000, -0.5000, -0.5000, -0.5000, -0.5000, -0.5000,
            -0.5000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000])

        self.log_level = "DEBUG"

        self.dataset_file_name = "dataset.csv"

        self.model_to_load_raw_weights = ""

        self.store_gradients_epoch_interval = 5

        self.check_test_loss_epoch_interval = 10

        self.dataset_normalization_mean = [0.4805, 0.3967, 0.3589]
        self.dataset_normalization_std = [0.3207, 0.2930, 0.2824]
        self.train_percentage = 80
        self.valid_percentage = 10
        self.test_percentage = 10
        self.paths_output_folder = ""
        self.paths_log_folder = ""
        self.paths_tensorboard_log_folder = ""
        self.paths_dataset = ""
        self.paths_train_many_models = ""
        self.paths_test_many_models = ""
        self.paths_tags = ""
        self.paths_graph = ""
        self.ensemble_model_configs = None
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
        if self.gcn_edge_index is not None and not isinstance(self.gcn_edge_index, torch.Tensor):
            self.gcn_edge_index = torch.tensor(self.gcn_edge_index)

        if self.gcn_edge_weights is not None and not isinstance(self.gcn_edge_weights, torch.Tensor):
            self.gcn_edge_weights = torch.tensor(self.gcn_edge_weights)

        if self.ensemble_model_configs is not None:
            self.ensemble_model_configs = [Config.from_dict(ensemble_config_data, default_config) for ensemble_config_data in self.ensemble_model_configs]

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