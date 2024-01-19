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
config = Config("default_config.yml")
logger = LoggerFactory.setup_logging("logger", log_file=pathutils.combine_path(
    pathutils.get_log_dir_path(config), 
    f"{config.model_name}_{config.model_image_size}_{config.model_weights}",
    f"train__{pathutils.get_datetime()}.log"), config=config)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_path = Path(args.input_path)
    output_folder = args.output_folder or input_path.parent.joinpath('inference_outputs')

    os.makedirs(output_folder, exist_ok=True)
    modelEvaluator = ModelEvaluator.from_file(device, config=config)

    if input_path.is_dir():
        image_paths = [os.path.join(input_path, img) for img in os.listdir(input_path)
                            if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_dataset = ImageDatasetPredict(image_paths, config=config)
        dataset_loader = DataLoader(image_dataset, batch_size=config.batch_size, shuffle=False)
        predictionResults = modelEvaluator.predict(dataset_loader, False, 0.5)
        predictions = predictionResults['predictions']
        image_paths = predictionResults['image_paths']
        flattened_image_paths = [path for sublist in image_paths for path in sublist]

        # Save the images with overlaid predictions
        for image_path, pred in zip(flattened_image_paths, predictions):
            original_image = Image.open(image_path)
            annotated_image = imageutils.overlay_predictions(original_image, pred, datasetutils.get_index_to_tag_mapping(config))
            save_path = os.path.join(output_folder, os.path.basename(image_path))
            annotated_image.save(save_path)
    elif input_path.is_file():
        if str(input_path).lower().endswith(('.png', '.jpg', '.jpeg')):
            preprocessed_img = ImageDatasetPredict.preprocess_single_image(str(input_path), config)
            predicted_labels = modelEvaluator.single_image_prediction(preprocessed_img, 0.5)
            original_image = Image.open(input_path)
            annotated_image = imageutils.overlay_predictions(original_image, predicted_labels, datasetutils.get_index_to_tag_mapping(config))
            save_path = os.path.join(output_folder, os.path.basename(input_path))
            annotated_image.save(save_path)

        elif str(input_path).lower().endswith(('.mp4', '.avi', '.mov')):
            input_path = str(input_path)
            video_dataset = VideoDatasetPredict(input_path, args.time_interval, config=config)
            dataset_loader = DataLoader(video_dataset, batch_size=config.batch_size, shuffle=False)
            predictionResults = modelEvaluator.predict(dataset_loader, False, 0.5)
            predictions = predictionResults['predictions']
            frame_counts = predictionResults['frame_counts']

            flattened_frame_counts = [frame_count for sublist in frame_counts for frame_count in sublist]

            save_path = os.path.join(output_folder, os.path.basename(input_path))
            imageutils.overlay_predictions_video(input_path, predictions, flattened_frame_counts, datasetutils.get_index_to_tag_mapping(config), save_path)
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
