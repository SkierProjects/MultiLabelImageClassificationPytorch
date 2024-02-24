import imclaslib.models.modelfactory as modelfactory
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch
import gc
import os
import imclaslib.files.pathutils as pathutils
from imclaslib.logging.loggerfactory import LoggerFactory
import imclaslib.models.modelutils as modelutils
import imclaslib.files.modelloadingutils as modelloadingutils
from torch.cuda.amp import autocast, GradScaler
import copy
import random
import wandb
logger = LoggerFactory.get_logger(f"logger.{__name__}")

class ModelTrainer():
    def __init__(self, device, trainloader, validloader, testloader, config, wandbWriter=None):
        """
        Initializes the ModelTrainer with the given datasets, device, and configuration.

        Parameters:
            device (torch.device): The device on which to train the model.
            trainloader (DataLoader): DataLoader for the training dataset.
            validloader (DataLoader): DataLoader for the validation dataset.
            testloader (DataLoader): DataLoader for the test dataset.
            config (module): Configuration module with necessary attributes.
        """
        self.config = config
        self.wandbWriter = wandbWriter
        self.metrics_enabled = (wandbWriter != None)
        self.device = device
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.model = modelfactory.create_model(self.config).to(device)
        if self.config.train_l2_enabled:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.train_learning_rate, weight_decay=self.config.train_l2_lambda)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.train_learning_rate)

        # Compute label frequencies and create weights for the loss function
        #self.label_freqs = self.compute_label_frequencies()
        #self.pos_weight = self.compute_loss_weights(self.label_freqs).to(device)
        self.criterion = nn.BCEWithLogitsLoss()#pos_weight=self.pos_weight)
        self.epochs = self.config.train_num_epochs
        self.lr_scheduler = modelutils.get_learningRate_scheduler(self.optimizer, config)
        self.last_train_loss = 10000
        self.last_valid_loss = 10000
        self.last_valid_f1 = 0
        self.current_lr = self.config.train_learning_rate

        modelToLoadPath = pathutils.get_model_to_load_path(self.config)
        if self.config.train_continue_training and os.path.exists(modelToLoadPath):
            logger.info("Loading the best model...")    
            if self.config.model_embedding_layer_enabled or self.config.model_gcn_enabled and self.config.train_model_to_load_raw_weights != "":
                self.model, modelData = modelloadingutils.load_pretrained_weights_exclude_classifier(self.model, self.config, True)
            else:
                modelData = modelloadingutils.load_model(modelToLoadPath, self.config)
                self.model.load_state_dict(modelData['model_state_dict'])
                #self.optimizer.load_state_dict(modelData['optimizer_state_dict'])

            self.start_epoch = modelData["epoch"] + 1
            self.epochs = self.epochs + self.start_epoch
            self.best_f1_score = 0.0
            self.__set_best_model_state(modelData["epoch"])

            
        else:
            self.best_f1_score = 0.0
            self.start_epoch = 0
            self.best_model_state = None
            self.__set_best_model_state(self.start_epoch)
        self.current_epoch = self.start_epoch - 1
        self.best_f1_score_at_last_reset = 0
        self.patience_counter = 0
        self.patience = self.config.train_early_stopping_patience
        
        if config.using_wsl and config.train_compile:
            self.compile()
        if self.metrics_enabled:
            self.wandbWriter.watch(self.model)
    
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
        del self.model
        del self.optimizer
        torch.cuda.empty_cache()
        gc.collect()
    
    def smooth_labels(self, labels):
        """
        Applies label smoothing. Turning the vector of 0s and 1s into a vector of
        `smoothing / num_classes` and `1 - smoothing + (smoothing / num_classes)`.
        Args:
            labels: The binary labels (0 or 1).
            smoothing: The degree of smoothing (0 means no smoothing).
        Returns:
            The smoothed labels.
        """
        smoothing = self.config.train_label_smoothing
        with torch.no_grad():
            num_classes = labels.size(1)
            # Create a tensor of `smoothing / num_classes` for each label
            smooth_value = smoothing / num_classes
            # Subtract smoothing from the 1s, add it to the 0s
            labels = labels * (1 - smoothing) + (1 - labels) * smooth_value
        return labels
    
    def compile(self):
        self.model = torch.compile(self.model)

    def train(self):
        """
        Train the model for one epoch using the provided training dataset.
        :return: The average training loss for the epoch.
        """
        self.current_epoch += 1
        logger.info('Training')
        self.model.train()
        train_running_loss = 0.0

        # Initialize the gradient scaler for mixed precision
        scaler = GradScaler(enabled=self.config.model_fp16)

        for data in tqdm(self.trainloader, total=len(self.trainloader)):
            images, targets = data['image'].to(self.device), data['label'].to(self.device).float()
            self.optimizer.zero_grad()

            # Cast operations to mixed precision
            with autocast(enabled=self.config.model_fp16):
                if (self.config.model_embedding_layer_enabled or self.config.model_gcn_enabled):
                    label_dropout_rate = 0.9
                    use_labels = random.random() > label_dropout_rate
                    if use_labels:
                        outputs = self.model(images, targets)
                    else:
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)

                # Verify that outputs and targets have the same shape
                if outputs.shape != targets.shape:
                    logger.error(f"Mismatched shapes detected: Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
                    # Here you could also raise an exception or handle the error in some way
                loss = self.criterion(outputs, self.smooth_labels(targets))

            # Scale the loss and call backward() to create scaled gradients
            scaler.scale(loss).backward()

            # Step optimizer and update the scale for next iteration
            scaler.step(self.optimizer)
            scaler.update()

            train_running_loss += loss.item()
        
        train_loss = train_running_loss / len(self.trainloader.dataset)
        self.last_train_loss = train_loss
        return train_loss
    
    def validate(self, modelEvaluator, threshold=None):
        """
        Validate the model on the validation dataset using a model evaluator.

        Parameters:
            modelEvaluator: An instance of the model evaluator class with an 'evaluate' method.
            threshold (Optional[float]): Threshold value for converting probabilities to class labels.

        Returns:
            tuple: A tuple containing the average validation loss and the micro-averaged F1 score.
        """
        logger.info("Validating")
        valid_loss, valid_f1, _, _ = modelEvaluator.evaluate(self.validloader, self.current_epoch, "Validation", threshold=threshold)
        self.last_valid_loss = valid_loss
        self.last_valid_f1 = valid_f1
        self.log_train_validation_results()
        return valid_loss, valid_f1
    
    def learningRateScheduler_check(self):
        """
        Check and update the learning rate based on the validation loss. Log the updated learning rate to TensorBoard.
        """
        self.lr_scheduler.step(self.last_valid_f1)
        oldlr = self.current_lr
        self.current_lr = self.optimizer.param_groups[0]['lr']
        if self.metrics_enabled:
            self.wandbWriter.log({"Train/Learning_Rate": self.current_lr}, step=self.current_epoch)
        if oldlr != self.current_lr:
            logger.info(f"Reducing learning rate from {oldlr} to {self.current_lr}")

    def log_train_validation_results(self):
        """
        Log training and validation results to the logger and TensorBoard.
        Includes the train loss, validation loss, and validation F1 score for the current epoch.
        """
        logger.info(f"Train Loss: {self.last_train_loss:.4f}")
        logger.info(f'Validation Loss: {self.last_valid_loss:.4f}')
        logger.info(f'Validation F1 Score: {self.last_valid_f1:.4f}')
        
        if self.metrics_enabled:
            self.wandbWriter.log({"Loss/Train": self.last_train_loss, "Loss/Validation": self.last_valid_loss, "F1/Validation": self.last_valid_f1}, step=self.current_epoch)
        
    def check_early_stopping(self):
        """
        Check if early stopping criteria are met based on the validation F1 score.
        If the score has not improved by a certain proportion over the patience window,
        trigger early stopping.

        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        improvement_threshold = self.config.train_early_stopping_threshold
        significant_improvement = False
        if self.last_valid_f1 > self.best_f1_score:
            logger.info(f"Validation F1 Score improved from {self.best_f1_score:.4f} to {self.last_valid_f1:.4f}")
            self.best_f1_score = self.last_valid_f1
            self.__set_best_model_state(self.current_epoch)

            modelloadingutils.save_best_model(self.best_model_state, self.config)

            # Check for significant improvement since the last reset of the patience counter
        if self.last_valid_f1 - self.best_f1_score_at_last_reset >= improvement_threshold:
            logger.info(f"Significant cumulative improvement of {self.last_valid_f1 - self.best_f1_score_at_last_reset:.4f} has been achieved since the last reset.")
            significant_improvement = True
            self.best_f1_score_at_last_reset = self.last_valid_f1
            self.patience_counter = 0
        
            # Increment patience counter if no significant improvement
        if not significant_improvement:
            self.patience_counter += 1

        # If there hasn't been significant improvement over the patience window, trigger early stopping
        if self.patience_counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.patience} epochs without significant cumulative improvement.")
            return True

    
    def save_final_model(self):
        """
        Save the state of the model that achieved the best validation F1 score during training.
        The model state is saved to a file defined by the configuration.
        """
        state_to_save = copy.deepcopy(self.best_model_state)
        modelloadingutils.save_final_model(self.best_model_state, self.best_f1_score, self.config)
        self.model.load_state_dict(state_to_save['model_state_dict'])

    def compute_label_frequencies(self):
        """
        Computes the frequency of each label in the dataset.
        
        Returns:
            label_freqs (torch.Tensor): Tensor containing the frequency of each label.
        """
        # Initialize a tensor to hold the frequency of each label.
        # This assumes that the number of labels is known and stored in `self.config.num_classes`.
        label_freqs = torch.zeros(self.config.model_num_classes, dtype=torch.float)

        # Iterate over the dataset and sum the one-hot encoded labels.
        for batch in tqdm(self.trainloader, total=len(self.trainloader)):
            labels = batch["label"]
            label_freqs += labels.sum(dim=0)  # Sum along the batch dimension.

        # Ensure that there's at least one count for each label to avoid division by zero.
        label_freqs = label_freqs.clamp(min=1) 
        return label_freqs
    
    def compute_loss_weights(self, label_freqs):
        """
        Computes the weights for each label to be used in the loss function.
        
        Parameters:
            label_freqs (torch.Tensor): Tensor containing the frequency of each label.
        
        Returns:
            weights (torch.Tensor): Tensor containing the weight for each label.
        """
        # Compute the inverse frequency weights
        total_counts = label_freqs.sum()
        weights = total_counts / label_freqs
        
        # Normalize weights to prevent them from scaling the loss too much
        weights = weights / weights.mean()

        #weights = weights.view(-1)  # Ensure it is a 1D tensor with shape [num_classes]
        #assert weights.shape[0] == self.config.num_classes, "pos_weight must have the same size as num_classes"
        
        return weights
    def __set_best_model_state(self, epoch):
        self.best_model_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion,
            'f1_score': self.best_f1_score,
            'model_name': self.config.model_name,
            'requires_grad': self.config.train_requires_grad,
            'model_num_classes': self.config.model_num_classes,
            'dropout': self.config.train_dropout_prob,
            'embedding_layer': self.config.model_embedding_layer_enabled,
            'model_gcn_enabled': self.config.model_gcn_enabled,
            'train_batch_size': self.config.train_batch_size,
            'optimizer': 'Adam',
            'loss_function': 'BCEWithLogitsLoss',
            'image_size':  self.config.model_image_size,
            'model_gcn_model_name': self.config.model_gcn_model_name,
            'model_gcn_out_channels': self.config.model_gcn_out_channels,
            'model_gcn_layers': self.config.model_gcn_layers,
            'model_attention_layer_num_heads': self.config.model_attention_layer_num_heads,
            'model_embedding_layer_dimension': self.config.model_embedding_layer_dimension,
            'train_loss': self.last_train_loss,
            'datset_version': self.config.dataset_version
        }