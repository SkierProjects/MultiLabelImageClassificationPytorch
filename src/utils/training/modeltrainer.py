import utils.models.modelfactory as modelfactory
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import os
import utils.files.pathutils as pathutils
from src.utils.logging.loggerfactory import LoggerFactory
from src.config import config
import src.utils.models.modelutils as modelutils
from utils.tensorboard.tensorboardwriter import TensorBoardWriter
import utils.files.modelloadingutils as modelloadingutils
logger = LoggerFactory.get_logger(f"logger.{__name__}")
final_model_path_template = pathutils.combine_path(pathutils.get_output_dir_path(), '{model_name}_{image_size}_{f1_score:.4f}.pth')

class ModelTrainer():
    def __init__(self, device, trainloader, validloader, testloader, config=config):
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
        self.model = modelfactory.create_model(
            self.config.model_name,
            requires_grad=self.config.model_requires_grad,
            num_classes=self.config.num_classes,
            weights=self.config.model_weights
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.epochs = self.config.num_epochs
        self.lr_scheduler = modelutils.get_learningRate_scheduler(self.optimizer)
        self.last_train_loss = 10000
        self.last_valid_loss = 10000
        self.last_valid_f1 = 0
        self.current_lr = self.config.learning_rate
        # Initialize TensorBoard writer
        self.tensorBoardWriter = TensorBoardWriter()

        modelToLoadPath = pathutils.get_model_to_load_path(self.config)
        if self.config.continue_training and os.path.exists(modelToLoadPath):
            logger.info("Loading the best model...")    
            self.best_f1_score, epochs = modelloadingutils.load_model(modelToLoadPath, self.model,self.optimizer)
            self.start_epoch = epochs + 1
            self.epochs = self.epochs + self.start_epoch
            self.best_model_state = {
                'epoch': epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.criterion,
                'f1_score': self.best_f1_score,
                'model_name': self.config.model_name,
                'requires_grad': self.config.model_requires_grad,
                'num_classes': self.config.num_classes,
                'dropout': self.config.model_dropout_prob,
                'embedding_layer': self.config.embedding_layer_enabled,
                'gcn_enabled': self.config.gcn_enabled,
                'batch_size': self.config.batch_size,
                'optimizer': 'Adam',
                'loss_function': 'BCEWithLogitsLoss'
            }
        else:
            self.best_f1_score = 0.0
            self.start_epoch = 0
            self.best_model_state = None
        self.current_epoch = self.start_epoch - 1
        self.patience_counter = 0
        self.patience = self.config.early_stopping_patience
    
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
            outputs = self.model(images)
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
        hparams = {
            'model_name': self.config.model_name,
            'requires_grad': self.config.model_requires_grad,
            'num_classes': self.config.num_classes,
            'dropout': self.config.model_dropout_prob,
            'embedding_layer': self.config.embedding_layer_enabled,
            'gcn_enabled': self.config.gcn_enabled,
            'lr': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'optimizer': 'Adam',
            'loss_function': 'BCEWithLogitsLoss'
        }
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
        if self.current_epoch % 5 == 0:  # Choose an interval that makes sense for your training regimen.
            for name, param in self.model.named_parameters():
                self.tensorBoardWriter.add_histogram(f'Parameters/{name}', param, self.current_epoch)
                if param.grad is not None:
                    self.tensorBoardWriter.add_histogram(f'Gradients/{name}', param.grad, self.current_epoch)
        
    def check_early_stopping(self):
        """
        Check if early stopping criteria are met based on the validation F1 score.
        If the score has not improved for a set number of epochs, trigger early stopping.

        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        if self.last_valid_f1 > self.best_f1_score:
            logger.info(f"Validation F1 Score improved from {self.best_f1_score:.4f} to {self.last_valid_f1:.4f}")
            self.best_f1_score = self.last_valid_f1
            self.best_model_state = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.criterion,
                'f1_score': self.best_f1_score,
                'model_name': self.config.model_name,
                'requires_grad': self.config.model_requires_grad,
                'num_classes': self.config.num_classes,
                'dropout': self.config.model_dropout_prob,
                'embedding_layer': self.config.embedding_layer_enabled,
                'gcn_enabled': self.config.gcn_enabled,
                'batch_size': self.config.batch_size,
                'optimizer': 'Adam',
                'loss_function': 'BCEWithLogitsLoss'
            }

            modelloadingutils.save_best_model(self.best_model_state)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement.")
                return True
        return False
    
    def save_final_model(self):
        """
        Save the state of the model that achieved the best validation F1 score during training.
        The model state is saved to a file defined by the configuration.
        """
        modelloadingutils.save_final_model(self.best_model_state, self.best_f1_score, self.config)
        self.model.load_state_dict(self.best_model_state['model_state_dict'])