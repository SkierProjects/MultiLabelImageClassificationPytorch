import wandb


class WandbWriter():
    """
    Initializes the Wandb Writer with a given configuration.

    Parameters:
        config (module): Configuration module with necessary attributes.
    """
    def __init__(self, config):
        wandb.init(
            # set the wandb project where this run will be logged
            project=config.project_name,
            config={
                'model_name': config.model_name,
                'requires_grad': config.train_requires_grad,
                'model_num_classes': config.model_num_classes,
                'dropout': config.train_dropout_prob,
                'embedding_layer': config.model_embedding_layer_enabled,
                'model_gcn_enabled': config.model_gcn_enabled,
                'train_batch_size': config.train_batch_size,
                'optimizer': 'Adam',
                'loss_function': 'BCEWithLogitsLoss',
                'image_size':  config.model_image_size,
                'model_gcn_model_name': config.model_gcn_model_name,
                'model_gcn_out_channels': config.model_gcn_out_channels,
                'model_gcn_layers': config.model_gcn_layers,
                'model_attention_layer_num_heads': config.model_attention_layer_num_heads,
                'model_embedding_layer_dimension': config.model_embedding_layer_dimension,
                'datset_version': config.dataset_version,
                'l2': config.train_l2_enabled,
                'l2_lambda': config.train_l2_lambda,
                'label_smoothing': config.train_label_smoothing,
                'dataset_normalization_mean': config.dataset_normalization_mean,
                'dataset_normalization_std': config.dataset_normalization_std,
            }
        )
    
    def log(self, *args, step=None):
        if step != None:
            wandb.log(*args, step=step)
        else:
            wandb.log(*args)

    def log_table(self, table_name, columnNames, columnData, step=None):
        if step != None:
            wandb.log({table_name: wandb.Table(columns=columnNames, data=columnData)}, step=step)
        else:
            wandb.log({table_name: wandb.Table(columns=columnNames, data=columnData)})

    def watch(self, model):
        wandb.watch(model)
    def __enter__(self):
        """
        Enter the runtime context for the ModelTrainer object.
        Allows the ModelTrainer to be used with the 'with' statement, ensuring resources are managed properly.

        Returns:
            ModelTrainer: The instance with which the context was entered.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context for the ModelTrainer object.
        This method is called after the 'with' block is executed, and it ensures that the TensorBoard writer is closed.

        Parameters:
            exc_type: Exception type, if any exception was raised within the 'with' block.
            exc_value: Exception value, the exception instance raised.
            traceback: Traceback object with details of where the exception occurred.
        """
        wandb.finish()