import csv
import time
from imclaslib.config import Config
import imclaslib.files.pathutils as pathutils
import argparse
import os
import torch
from imclaslib.evaluation.modelevaluator import ModelEvaluator
from imclaslib.dataset.video_predict_dataset import VideoDatasetPredict
from imclaslib.dataset.images_predict_dataset import ImageDatasetPredict
from imclaslib.dataset import datasetutils
from imclaslib.files import imageutils
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from imclaslib.logging.loggerfactory import LoggerFactory
from imclaslib.metrics import metricutils
thisconfig = Config("default_config.yml")
logger = LoggerFactory.setup_logging("logger", log_file=pathutils.combine_path(thisconfig, 
    pathutils.get_log_dir_path(thisconfig), 
    f"{thisconfig.model_name}_{thisconfig.model_image_size}_{thisconfig.model_weights}",
    f"train__{pathutils.get_datetime()}.log"), config=thisconfig)
thisconfig = Config.load_config_from_file("inference.yml", thisconfig)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_path = Path(args.input_path)
    output_folder = args.output_folder or input_path.parent.joinpath('inference_outputs')

    os.makedirs(output_folder, exist_ok=True)
    if thisconfig.model_ensemble_model_configs != None:
        modelEvaluator = ModelEvaluator.from_ensemble(device, thisconfig=thisconfig)
    else:
        modelEvaluator = ModelEvaluator.from_file(device, thisconfig=thisconfig)
    if thisconfig.using_wsl and thisconfig.train_compile:
        modelEvaluator.compile()

    if input_path.is_dir():
        output_csv_path = "annotationresults.csv"
        image_paths = []
        for root, dirs, files in os.walk(input_path):
            for img in files:
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, img))
        image_dataset = ImageDatasetPredict(image_paths, config=thisconfig)
        dataset_loader = DataLoader(image_dataset, batch_size=thisconfig.test_batch_size, shuffle=False, num_workers=6, persistent_workers=True, pin_memory=False)
        optimalTemp = thisconfig.model_temperature
        predict_start_time = time.time()
        predictionResults = modelEvaluator.predict(dataset_loader, False)
        predict_end_time = time.time()

        seconds = predict_end_time-predict_start_time
        image_count = len(image_dataset)
        logger.info(f"Took {seconds} seconds to predict for {image_count} images with an average of {image_count/seconds} images per second")
        logits = torch.Tensor(predictionResults['predictions']).to(device)
        scaled_logits = metricutils.temperature_scale(logits, optimalTemp)
        prediction_confidences = metricutils.getConfidences(scaled_logits)
        predictions = metricutils.getpredictions_with_threshold(scaled_logits, device, threshold=0.5)
        image_paths = predictionResults['image_paths']
        flattened_image_paths = [path for sublist in image_paths for path in sublist]
        uncertainty_metrics = metricutils.uncertainty_metrics(prediction_confidences)
        cumul_uncertainty = uncertainty_metrics['cumulative_uncertainties']
        max_uncertainty = uncertainty_metrics['max_uncertainties']
        mean_uncertainty = uncertainty_metrics['mean_uncertainties']
        mean_entropy = uncertainty_metrics['mean_entropies']

        # Prepare CSV data
        csv_data = []
        for image_path, pred, cumul_uncertainty, max_uncert, mean_uncert, mean_entrop in zip(flattened_image_paths, predictions, cumul_uncertainty, max_uncertainty, mean_uncertainty, mean_entropy):
            # Make sure pred is a list and uncertainty is a scalar
            pred_list = pred.tolist() if isinstance(pred, torch.Tensor) else list(pred)
            uncertainty_scalar = float(cumul_uncertainty)  # Convert to a Python float
            # Create a row with image file name, cumulative uncertainty, and one-hot encoded predictions
            if thisconfig.using_wsl:
                image_path = pathutils.convert_wsl_path_to_windows(image_path)
            row = [image_path, uncertainty_scalar, max_uncert[0], max_uncert[1], mean_uncert, mean_entrop] + pred_list
            csv_data.append(row)

        while True:
            try:
                # Write the CSV data to a file
                with open(output_csv_path, 'w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    # Write the header
                    tagmappings = datasetutils.get_index_to_tag_mapping(thisconfig)
                    header = ['file_name', 'cumulative_uncertainty', 'Max Uncertainty', 'Max Uncertainty Tag', 'Mean Uncertainty', 'Mean Entropy'] + [f'{tagmappings[i]}' for i in range(len(predictions[0]))]
                    csv_writer.writerow(header)
                    # Write the rows
                    csv_writer.writerows(csv_data)
                break  # Exit the loop if file writing was successful
            except PermissionError:
                print("The file is currently open and cannot be written to. Please close the file and press Enter to retry.")
                input()  # Wait for user to indicate they've closed the file
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break  # Exit the loop if an unexpected error occurs


        #Save the images with overlaid predictions
        for image_path, pred in zip(flattened_image_paths, predictions):
           original_image = Image.open(image_path)
           annotated_image = imageutils.overlay_predictions(original_image, pred, datasetutils.get_index_to_tag_mapping(thisconfig))
           save_path = os.path.join(output_folder, os.path.basename(image_path))
           annotated_image.save(save_path)
    elif input_path.is_file():
        if str(input_path).lower().endswith(('.png', '.jpg', '.jpeg')):
            preprocessed_img = ImageDatasetPredict.preprocess_single_image(str(input_path), thisconfig)
            predicted_labels = modelEvaluator.single_image_prediction(preprocessed_img, 0.5)
            original_image = Image.open(input_path)
            annotated_image = imageutils.overlay_predictions(original_image, predicted_labels, datasetutils.get_index_to_tag_mapping(thisconfig))
            save_path = os.path.join(output_folder, os.path.basename(input_path))
            annotated_image.save(save_path)

        elif str(input_path).lower().endswith(('.mp4', '.avi', '.mov')):
            input_path = str(input_path)
            video_dataset = VideoDatasetPredict(input_path, args.time_interval, config=thisconfig)
            dataset_loader = DataLoader(video_dataset, batch_size=thisconfig.test_batch_size, shuffle=False)
            predictionResults = modelEvaluator.predict(dataset_loader, False, 0.5)
            predictions = predictionResults['predictions']
            frame_counts = predictionResults['frame_counts']

            flattened_frame_counts = [frame_count for sublist in frame_counts for frame_count in sublist]

            save_path = os.path.join(output_folder, os.path.basename(input_path))
            imageutils.overlay_predictions_video(input_path, predictions, flattened_frame_counts, datasetutils.get_index_to_tag_mapping(thisconfig), save_path)
        else:
            print(f"Unsupported file type for input: {input_path}")
    else:
        print(f"Invalid input path: {input_path}")

# Define the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on images or videos.')
    parser.add_argument('input_path', type=str, help='Path to an input image, directory of images, or video file.')
    parser.add_argument('--output_folder', type=str, help='Path to save the output predictions.', default=None)
    parser.add_argument('--time_interval', type=float, help='Interval in seconds of how frequently to process frames from a video.', default=2)

    args = parser.parse_args()
    main(args)
