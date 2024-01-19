from torch.utils.tensorboard import SummaryWriter
import imclaslib.files.pathutils as pathutils
import imclaslib.files.imageutils as imageutils
import imclaslib.dataset.datasetutils as datasetutils

class TensorBoardWriter():
    """
    Initializes the TensorBoardWriter with a given configuration.

    Parameters:
        config (module): Configuration module with necessary attributes.
    """
    def __init__(self, config):
        self.config = config

        modelAddons = ""
        if self.config.embedding_layer_enabled:
            modelAddons = f"_EmbeddingLayer_{config.embedding_layer_dimension}"
        elif self.config.gcn_enabled:
            modelAddons = f"_GCN_{config.embedding_layer_dimension}_{config.gcn_out_channels}_{config.gcn_layers}_{config.attention_layer_num_heads}"
        log_dir = pathutils.combine_path(
            pathutils.get_tensorboard_log_dir_path(),
            f'{config.model_name}_{config.model_weights}_{config.model_image_size}_{config.model_dropout_prob}{modelAddons}'
        )
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag, scalar_value, step):
        """
        Writes a scalar value to TensorBoard.

        Parameters:
            tag (str): The tag associated with the scalar.
            scalar_value (float): The scalar value to write.
            step (int): The global step value to record.
        """
        self.writer.add_scalar(tag, scalar_value, step)

    def add_scalars_from_dict(self, input_dict, step):
        """
        Writes a scalar value to TensorBoard.

        Parameters:
            input_dict (dict): Dictionary of tag name to tag values
            step (int): The global step value to record.
        """
        for key, value in input_dict.items():
            self.add_scalar(key, value, step)

    def write_image_test_results(self, images, true_labels, predictions, step, runmode, dataSubset):
        """
        Writes image test results with overlays to TensorBoard.

        Parameters:
            images (Tensor): Batch of images.
            true_labels (Tensor): True labels for the images.
            predictions (Tensor): Predicted labels for the images.
            step (int): The global step value to record.
            runmode (str): The mode of the run (e.g., 'Train', 'Test').
            data_subset (str): The subset of data (e.g., 'Validation').
        """
        denormalized_images = imageutils.denormalize_images(images, self.config)
        pil_images = imageutils.convert_to_PIL(denormalized_images)
        overlaid_images = imageutils.overlay_predictions_batch(pil_images, predictions.cpu().tolist(), datasetutils.get_index_to_tag_mapping(), true_labels.cpu().tolist())
        tensor_overlaid_images = imageutils.convert_PIL_to_tensors(overlaid_images)
        self.add_images(f'{runmode}/{dataSubset}/Images', denormalized_images, step)
        self.add_images(f'{runmode}/{dataSubset}/True Labels', imageutils.convert_labels_to_color(true_labels.cpu(), self.config.num_classes), step)
        self.add_images(f'{runmode}/{dataSubset}/Predictions', imageutils.convert_labels_to_color(predictions.cpu(), self.config.num_classes), step)
        self.add_images(f'{runmode}/{dataSubset}/OverlayPredictions', tensor_overlaid_images, step)

    def add_histogram(self, tag, param, step):
        """
        Writes a histogram of values to TensorBoard.

        Parameters:
            tag (str): The tag associated with the histogram.
            values (Tensor): Values to create a histogram.
            step (int): The global step value to record.
        """
        self.writer.add_histogram(tag, param, step)

    def add_images(self, tag, images, step):
        """
        Writes a batch of images to TensorBoard.

        Parameters:
            tag (str): The tag associated with the images.
            images (Tensor): Batch of images to write.
            step (int): The global step value to record.
        """
        self.writer.add_images(tag, images, step)

    def close_writer(self):
        """
        Closes the TensorBoard writer and cleans up resources.
        """
        if self.writer:
            self.writer.close()
            self.writer = None

    def add_hparams(self, hparams, metrics):
        """
        Writes hyperparameters and their associated metrics to TensorBoard.

        Parameters:
            hparams (dict): Dictionary of hyperparameters.
            metrics (dict): Dictionary of metrics associated with the hyperparameters.
        """
        self.writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)