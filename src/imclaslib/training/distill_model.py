import time
from imclaslib.logging.loggerfactory import LoggerFactory
logger = LoggerFactory.get_logger(f"logger.{__name__}")

import torch
import imclaslib.dataset.datasetutils as datasetutils
from imclaslib.training.modeltrainer import ModelTrainer
from imclaslib.evaluation.modelevaluator import ModelEvaluator
from imclaslib.evaluation.test_model import evaluate_model

def distill_model(teacher_config, student_config, wandbWriter=None):
    """
    Train a model based on the provided configuration.

    Parameters:
        config: Configuration module with necessary attributes.
    """
    logger.info(f"Teacher image size:{teacher_config.model_image_size}, Student image size:{student_config.model_image_size}")
    # Initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get train, validation, and test dataset loaders
    teacher_train_loader = datasetutils.get_data_loader_by_name('train', config=teacher_config)
    train_loader, valid_loader, test_loader = datasetutils.get_train_valid_test_loaders(config=student_config)
    try:
        # Initialize the model trainer 
        with ModelTrainer(device, train_loader, valid_loader, test_loader, config=student_config, wandbWriter=wandbWriter) as modelTrainer, ModelEvaluator.from_trainer(modelTrainer) as modelEvaluator, ModelEvaluator.from_file(device, teacher_config) as teacherEvaluator:
            # Start the training and validation
            try:
                for epoch in range(modelTrainer.start_epoch, modelTrainer.epochs):
                    logger.info(f"Epoch {epoch+1} of {modelTrainer.epochs}")
                    
                    # Training and validation steps
                    train_start_time = time.time()
                    modelTrainer.distill(teacherEvaluator.model, teacher_train_loader)
                    train_end_time = time.time()
                    logger.info(f"Finished Training Epoch in {train_end_time - train_start_time} seconds.")
                    modelTrainer.validate(modelEvaluator)

                    # Check for early stopping
                    if modelTrainer.check_early_stopping():
                        break

                    # Update learning rate based on validation loss
                    modelTrainer.learningRateScheduler_check()
                    
                    # Evaluate test results at specified intervals
                    if epoch % student_config.train_check_test_loss_epoch_interval == 0 and epoch != 0:
                        logger.info("Evaluating Test Results")
                        test_loss, test_f1, precision, recall = modelEvaluator.evaluate(test_loader, epoch, "Test", "Train")
                        logger.info(f'Test Loss: {test_loss:.4f}')
                        logger.info(f'Test Precision: {precision:.4f}, Recall:{recall:.4f}')
                        logger.info(f'Test F1 Score: {test_f1:.4f}')
            except KeyboardInterrupt:
                logger.warn("\nTraining interrupted by user.")
            except Exception as e:
                 logger.error('Error at %s', 'distill training loop', exc_info=e)
                 raise e
            finally:
                if modelTrainer.best_model_state:
                    modelTrainer.save_final_model()
    except Exception as e:
        raise e
    finally:
        student_config.model_name_to_load = "best_model"
        evaluate_model(student_config, valid_loader, test_loader, wandbWriter=wandbWriter)
        return train_loader, valid_loader, test_loader