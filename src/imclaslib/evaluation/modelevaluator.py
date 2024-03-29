# evaluator.py
import torch
import wandb
from imclaslib.logging.loggerfactory import LoggerFactory
import imclaslib.files.pathutils as pathutils
import imclaslib.models.modelfactory as modelfactory
import imclaslib.metrics.metricutils as metricutils
import imclaslib.files.modelloadingutils as modelloadingutils
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import random
import os
import itertools
import gc

# Initialize logger for this module.
logger = LoggerFactory.get_logger(f"logger.{__name__}")

class ModelEvaluator:
    def __init__(self, model, criterion, device, config, wandbWriter=None, model_data=None):
        """
        Initializes the ModelEvaluator with a given model, loss criterion, device,
        optional TensorBoard writer, and configuration.

        Parameters:
            model (torch.nn.Module): The model to evaluate.
            criterion (function): The loss function.
            device (torch.device): The device to run evaluation on (CPU or GPU).
            wandbWriter (WandbWriter, optional): Writer for Wandb logging.
            config (object): An immutable configuration object with necessary parameters.
        """
        self.config = config
        self.model = model
        self.criterion = criterion
        self.device = device
        self.num_classes = config.model_num_classes
        self.wandbWriter = wandbWriter
        self.model_data = model_data
        self.metrics_enabled = (wandbWriter != None)

    def __enter__(self):
        """
        Context management method to use with 'with' statements.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context management method to close the TensorBoard writer upon exiting the 'with' block.
        """
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

    @classmethod
    def from_trainer(cls, model_trainer):
        """
        Creates a ModelEvaluator instance from a ModelTrainer instance by extracting
        the relevant attributes.

        Parameters:
            model_trainer (ModelTrainer): The trainer instance to extract attributes from.

        Returns:
            ModelEvaluator: A new instance of ModelEvaluator.
        """
        return cls(
            model=model_trainer.model,
            criterion=model_trainer.criterion,
            device=model_trainer.device,
            config=model_trainer.config,
            wandbWriter=model_trainer.wandbWriter,
            model_data=model_trainer.best_model_state
        )
    
    @classmethod
    def from_file(cls, device, thisconfig, wandbWriter=None):
        """
        Creates a ModelEvaluator instance from a model file by loading in the model and preparing it
          to be run.

        Parameters:
            device (torch.device): The device to run evaluation on (CPU or GPU).
            wandbWriter (WandbWriter, optional): Writer for Wandb logging.
            config (object): An immutable configuration object with necessary parameters.
        """
        
        model = modelfactory.create_model(thisconfig).to(device)
        criterion = nn.BCEWithLogitsLoss()

        modelToLoadPath = pathutils.get_model_to_load_path(thisconfig)
        if os.path.exists(modelToLoadPath):
            logger.info("Loading the best model...")    
            modelData = modelloadingutils.load_model(modelToLoadPath, thisconfig)
            model.load_state_dict(modelData['model_state_dict'])
            logger.info("Loaded the best model in the Evaluator")
        else:
            logger.error(f"Could not find a model at path: {modelToLoadPath}")
            raise ValueError(f"Could not find a model at path: {modelToLoadPath}. Check to ensure the config/json value for model_name_to_load is correct!")
        
        return cls(
            model=model,
            criterion=criterion,
            device=device,
            config=thisconfig,
            wandbWriter=wandbWriter,
            model_data=modelData
        )
    
    @classmethod
    def from_ensemble(cls, device, thisconfig, wandbWriter=None, loadFromFile=False):
        """
        Creates a ModelEvaluator instance from a model file by loading in the model and preparing it
          to be run.

        Parameters:
            device (torch.device): The device to run evaluation on (CPU or GPU).
            wandbWriter (WandbWriter, optional): Writer for TensorBoard logging.
            config (object): An immutable configuration object with necessary parameters.
        """
        
        model = modelfactory.create_model(thisconfig).to(device)
        criterion = nn.BCEWithLogitsLoss()
        
        if loadFromFile:
            modelToLoadPath = pathutils.get_model_to_load_path(thisconfig)
            modelData = modelloadingutils.load_model(modelToLoadPath, thisconfig)
            model.load_state_dict(modelData['model_state_dict'])

        model_data = {
            "epoch": 1,
            "train_loss": 0,
        }
        
        return cls(
            model=model,
            criterion=criterion,
            device=device,
            config=thisconfig,
            wandbWriter=wandbWriter,
            model_data=model_data
        )
    
    def single_image_prediction(self, preprocessed_image, threshold=None):
        """Run a prediction for a single preprocessed image."""
        self.model.eval()  # Set the model to evaluation mode
        
        # Move the preprocessed image to the same device as the model
        preprocessed_image = preprocessed_image.to(self.device)
        
        with torch.no_grad():
            # Add a batch dimension to the image tensor
            image_batch = preprocessed_image.unsqueeze(0)
            outputs = self.model(image_batch)
            if threshold is not None:
                outputs_np = metricutils.getpredictions_with_threshold(outputs_np, self.device, threshold)
                # Wrap the NumPy array back into a PyTorch tensor if necessary
                outputs = torch.from_numpy(outputs_np)
            # Remove the batch dimension from the outputs before returning
            outputs = outputs.squeeze(0)
        return outputs
    
    def compile(self):
        self.model = torch.compile(self.model, mode="max-autotune")
    def predict(self, data_loader, return_true_labels=True, threshold=None):
        """
        Perform inference on the given data_loader and return raw predictions.

        Parameters:
            data_loader (DataLoader): DataLoader for inference.
            return_true_labels (bool): If true, return true labels. Otherwise, skip label processing.

        Returns:
            prediction_labels (numpy.ndarray): Raw model outputs.
            true_labels (numpy.ndarray, optional): Corresponding true labels, if available and requested.
            avg_loss (float, optional): Average loss over dataset, if labels are available.
        """
        model = self.model
        model.eval()  # Set the model to evaluation mode
        prediction_outputs = []  # List to store all raw model outputs
        true_labels = []  # List to store all labels if they are available
        image_paths = [] # List to store all image paths if they are available
        frame_counts = [] # List to store all frame counts if they are available
        total_loss = 0.0  # Initialize total loss
        with torch.no_grad():  # Disable gradient calculation for efficiency
            for batch in tqdm(data_loader, total=len(data_loader)):
                images = batch['image'].to(self.device)
                if self.config.model_fp16:
                    images = images.half()
                with autocast(enabled=self.config.model_fp16):
                    outputs = model(images)
                prediction_outputs.append(outputs.cpu().numpy())  # Store raw model outputs
                
                # Process labels if they are available and requested
                if return_true_labels and 'label' in batch:
                    labels = batch['label'].to(self.device)
                    loss = self.criterion(outputs, labels.float())  # Calculate loss
                    total_loss += loss.item()  # Accumulate loss
                    true_labels.append(labels.cpu().numpy())  # Store labels
                elif not return_true_labels and 'image_path' in batch:
                    image_paths.append(batch['image_path'])
                elif not return_true_labels and 'frame_count' in batch:
                    frame_counts.append(batch['frame_count'])
        # Concatenate all raw outputs and optionally labels from all batches
        prediction_outputs = np.vstack(prediction_outputs)
        results = {'predictions': prediction_outputs}
        
        if return_true_labels and true_labels:
            true_labels = np.vstack(true_labels)
            avg_loss = total_loss / len(data_loader.dataset)
            results['true_labels'] = true_labels
            results['avg_loss'] = avg_loss

        if image_paths:
            results['image_paths'] = image_paths

        if frame_counts:
            results['frame_counts'] = frame_counts

        if threshold != None:
            predictions_binary = metricutils.getpredictions_with_threshold(prediction_outputs, self.device, threshold)
            results['predictions'] = predictions_binary

        return results

    def evaluate_predictions(self, data_loader, prediction_outputs, true_labels, epoch, average, datasetSubset=None, metricMode=None, threshold=None):
        """
        Evaluate the model on the given data_loader.

        Parameters:
            data_loader (DataLoader): DataLoader for evaluation.
            prediction_outputs (numpy.ndarray): Raw model outputs.
            true_labels (numpy.ndarray): Corresponding true labels.
            epoch (int): The current epoch number, used for TensorBoard logging.
            datasetSubset (str): Indicates the subset of data evaluated (e.g., 'test', 'validation').
            average (str): Indicates the type of averaging to perform when computing metrics. Use None to get per-class metrics.
            metricMode (str, optional): Indicates from where this is being evaluated from (e.g., 'Train', 'Test').
            threshold (float, optional): The threshold value for binary predictions.

        Returns:
            f1_score (float): The F1 score of the model on the dataset.
            precision (float): The precision of the model on the dataset.
            recall (float): The recall of the model on the dataset.
        """
        predictions_binary = metricutils.getpredictions_with_threshold(prediction_outputs, self.device, threshold)
        # Compute evaluation metrics
        precision, recall, f1 = metricutils.compute_metrics(true_labels, predictions_binary, average=average)
        #if f1 >= 0.9:
            #something is wrong
        #    i = 1
        # Log images with predictions to TensorBoard for a random batch, if configured
        if metricMode is not None and self.wandbWriter is not None and datasetSubset is not None:
            random_batch_index = random.randint(0, len(data_loader) - 1)
            batch_dict = next(itertools.islice(data_loader, random_batch_index, None))
            images = batch_dict['image']  # Assuming the device transfer happens elsewhere if needed
            labels = batch_dict['label']
            
            start_index = random_batch_index * data_loader.batch_size
            end_index = min((random_batch_index + 1) * data_loader.batch_size, len(predictions_binary))

            selected_predictions = predictions_binary[start_index:end_index]
            selected_predictions_tensor = torch.tensor(selected_predictions, device=self.device, dtype=torch.float32)
            #self.tensorBoardWriter.write_image_test_results(images, labels, selected_predictions_tensor, epoch, metricMode, datasetSubset)
        # Return the average loss and computed metrics
        return f1, precision, recall

    def evaluate(self, data_loader, epoch, datasetSubset, metricMode=None, average='micro', threshold=None):
        """
        Evaluate the model on the given data_loader.

        Parameters:
            data_loader (DataLoader): DataLoader for evaluation.
            epoch (int): The current epoch number, used for TensorBoard logging.
            datasetSubset (str): Indicates the subset of data being evaluated (e.g., 'test', 'validation').
            average (str): Indicates the type of averaging to perform when computing metrics. Use None to get per-class metrics.
            metricMode (str, optional): Indicates from where this is being evaluated from (e.g., 'Train', 'Test').
            threshold (float, optional): The threshold value for binary predictions.

        Returns:
            avg_loss (float): The average loss over the dataset.
            f1_score (float): The F1 score of the model on the dataset.
            precision (float): The precision of the model on the dataset.
            recall (float): The recall of the model on the dataset.
        """
        # Perform inference and get raw outputs
        prediction_results = self.predict(data_loader)
        all_outputs, all_labels, avg_loss = prediction_results['predictions'], prediction_results['true_labels'], prediction_results['avg_loss']

        f1, precision, recall = self.evaluate_predictions(data_loader, all_outputs, all_labels, epoch, average, datasetSubset, metricMode, threshold)

        # Return the average loss and computed metrics
        return avg_loss, f1, precision, recall
