import torch

class config:
    """
    Configuration class for holding model and training parameters.
    """

    # Default static property values
    model_name = 'convnext_large'
    model_requires_grad = True
    num_classes = 31
    model_dropout_prob = 0
    model_weights = 'DEFAULT'
    image_size = 400
    batch_size = 24
    learning_rate = 1e-2
    num_epochs = 50
    continue_training = False
    model_name_to_load = "best_model"
    early_stopping_patience = 10
    early_stopping_threshold = 8e-3
    learningrate_reducer_patience = 2
    learningrate_reducer_threshold = 1e-2
    learningrate_reducer_factor = 0.1
    learningrate_reducer_min_lr = 1e-7

    embedding_layer_enabled = False
    embedding_layer_dimension = 512
    gcn_enabled = False
    gcn_model_name = "GAT"
    gcn_out_channels = 512
    gcn_layers = 4
    attention_layer_num_heads = 8
    # Define the edges
    gcn_edge_index = torch.tensor(
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
    gcn_edge_weights = torch.tensor(
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

    log_level = "DEBUG"

    dataset_file_name = "dataset.csv"

    model_to_load_raw_weights = ""

    store_gradients_epoch_interval = 5

    check_test_loss_epoch_interval = 10

    dataset_normalization_mean = [0.4805, 0.3967, 0.3589]
    dataset_normalization_std = [0.3207, 0.2930, 0.2824]
    train_percentage = 80
    valid_percentage = 10
    test_percentage = 10
    ensemble_model_configs = []

    def __init__(self, config_path=None):
        """
        Initialize a new Config instance, optionally loading values from a JSON file.

        Parameters:
        - config_path: str (optional), path to a JSON file with configuration values.
        """
        if config_path:
            self.load_from_json(config_path)

    @classmethod
    def load_from_json(cls, config_data):
        """
        Load configuration data from a dictionary, typically loaded from a JSON file,
        and update the configuration instance.

        Parameters:
        - config_data: dict, dictionary containing configuration keys and values.

        Returns:
        - Config instance with updated values.
        """
        new_instance = cls()  # Create a new instance with default values
        for key, value in config_data.items():
            normalized_key = key.lower()  # Normalize the key to lowercase
            if normalized_key == 'ensemble_model_configs':
                # Recursively load ensemble models
                new_instance.ensemble_model_configs = [cls.load_from_json(ensemble_config_data) for ensemble_config_data in value]
            elif hasattr(new_instance, normalized_key):
                setattr(new_instance, normalized_key, value)
        return new_instance
