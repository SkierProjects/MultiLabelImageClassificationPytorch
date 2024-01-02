class config:
    """
    Configuration class for holding model and training parameters.
    """

    # Default static property values
    model_name = 'resnext50_32x4d'
    model_requires_grad = True
    num_classes = 31
    model_dropout_prob = 0.0
    model_weights = 'DEFAULT'
    image_size = 384
    batch_size = 48
    learning_rate = 1e-4
    num_epochs = 150
    continue_training = False
    model_name_to_load = "best_model"
    early_stopping_patience = 8
    learningrate_reducer_patience = 3
    learningrate_reducer_threshold = 1e-2
    learningrate_reducer_factor = 0.1

    embedding_layer_enabled = False
    gcn_enabled = False

    log_level = "DEBUG"

    dataset_file_name = "dataset.csv"

    store_gradients_epoch_interval = 5

    check_test_loss_epoch_interval = 10

    dataset_normalization_mean = [0.4805, 0.3967, 0.3589]
    dataset_normalization_std = [0.3207, 0.2930, 0.2824]


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
            if hasattr(new_instance, normalized_key):
                setattr(new_instance, normalized_key, value)
        return new_instance