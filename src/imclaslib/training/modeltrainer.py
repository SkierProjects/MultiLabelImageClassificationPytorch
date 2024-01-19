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
from imclaslib.metrics import metricutils
from imclaslib.tensorboard.tensorboardwriter import TensorBoardWriter
import imclaslib.files.modelloadingutils as modelloadingutils
import copy
import random
logger = LoggerFactory.get_logger(f"logger.{__name__}")

class ModelTrainer():
    def __init__(self, device, trainloader, validloader, testloader, config):
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
        self.device = device
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.model = modelfactory.create_model(self.config).to(device)
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
        # Initialize TensorBoard writer
        self.tensorBoardWriter = TensorBoardWriter(config)

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
        self.current_epoch = self.start_epoch - 1
        self.best_f1_score_at_last_reset = 0
        self.patience_counter = 0
        self.patience = self.config.train_early_stopping_patience
    
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
        self.tensorBoardWriter.close_writer()
        del self.model
        del self.optimizer
        torch.cuda.empty_cache()
        gc.collect()
    
    def train(self):
        """
        Train the model for one epoch using the provided training dataset.
        :return: The average training loss for the epoch.
        """
        self.current_epoch += 1
        logger.info('Training')
        self.model.train()
        train_running_loss = 0.0
        for data in tqdm(self.trainloader, total=len(self.trainloader)):
            images, targets = data['image'].to(self.device), data['label'].to(self.device).float()
            self.optimizer.zero_grad()

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
            loss = self.criterion(outputs, targets)
            train_running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        
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
        self.lr_scheduler.step(self.last_valid_loss)
        self.current_lr = self.optimizer.param_groups[0]['lr']
        self.tensorBoardWriter.add_scalar('Learning Rate', self.current_lr, self.current_epoch)

    def log_train_validation_results(self):
        """
        Log training and validation results to the logger and TensorBoard.
        Includes the train loss, validation loss, and validation F1 score for the current epoch.
        """
        logger.info(f"Train Loss: {self.last_train_loss:.4f}")
        logger.info(f'Validation Loss: {self.last_valid_loss:.4f}')
        logger.info(f'Validation F1 Score: {self.last_valid_f1:.4f}')
        
        self.tensorBoardWriter.add_scalar('Loss/Train', self.last_train_loss, self.current_epoch)
        self.tensorBoardWriter.add_scalar('Loss/Validation', self.last_valid_loss, self.current_epoch)
        self.tensorBoardWriter.add_scalar('F1/Validation', self.last_valid_f1, self.current_epoch)

    def log_hparam_results(self, test_loss, test_f1):
        """
        Log the hyperparameters and test metrics to TensorBoard.
        This method is used for visualizing the relationship between hyperparameters and the model's performance.

        Parameters:
            test_loss (float): The loss on the test dataset.
            test_f1 (float): The F1 score on the test dataset.
        """
        hparams = metricutils.filter_dict_for_hparams(self.best_model_state)
        metrics = {
            'best_val_f1_score': self.best_f1_score,
            'final_train_loss': self.last_train_loss if self.last_train_loss else 0,
            'final_valid_loss': self.last_valid_loss if self.last_valid_loss else 0,
            'test_f1_score': test_f1,
            'test_loss': test_loss
        }
        self.tensorBoardWriter.add_hparams(hparams, metrics)

    def log_gradients(self):
        """
        Log the gradients of model parameters to TensorBoard.
        This is done periodically based on the current epoch to monitor training progress and diagnose issues.
        """
        if self.current_epoch % self.config.train_store_gradients_epoch_interval == 0:  # Choose an interval that makes sense for your training regimen.
            for name, param in self.model.named_parameters():
                self.tensorBoardWriter.add_histogram(f'Parameters/{name}', param, self.current_epoch)
                if param.grad is not None:
                    self.tensorBoardWriter.add_histogram(f'Gradients/{name}', param.grad, self.current_epoch)
        
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
            'train_loss': self.last_train_loss
        }