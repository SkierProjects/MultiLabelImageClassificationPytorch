import torchvision.transforms as transforms
import torch
from PIL import ImageDraw, Image, ImageFont
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import cv2
from torchvision.transforms.functional import to_pil_image
import numpy as np

from imclaslib.metrics import metricutils

def denormalize_images(images, config):
    """
    Denormalizes a batch of images using the specified mean and standard deviation.

    Parameters:
        images (torch.Tensor): Batch of images to denormalize.
        config (object): Configuration object with mean and std for denormalization.

    Returns:
        torch.Tensor: Batch of denormalized images.
    """
    return torch.stack([denormalize(img.cpu(), config.dataset_normalization_mean, config.dataset_normalization_std) for img in images])

def denormalize(tensor, mean, std):
    """De-normalizes a tensor image with mean and standard deviation."""
    # Clone the tensor so we don't change the original
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # De-normalize
    return torch.clamp(tensor, 0, 1)

def overlay_predictions_batch(images, predictions, index_to_tag, true_labels=None):
    """
    Overlays prediction and ground truth labels on images.

    Parameters:
        images (list or torch.Tensor): Batch of images to annotate.
        true_labels (torch.Tensor): True labels for each image.
        predictions (torch.Tensor): Predicted labels for each image.
        index_to_tag (dict): Mapping from label indices to tag names.

    Returns:
        list: List of annotated images.
    """
    annotated_images = []
    for img, true_label_vec, pred_label_vec in zip(images, true_labels, predictions):
        annotated_images.append(overlay_predictions(img, pred_label_vec, index_to_tag, true_label_vec))

    return annotated_images

def overlay_predictions(image, predictions, index_to_tag, true_labels=None):
    """
    Overlays prediction and ground truth labels on images.

    Parameters:
        image (PIL.Image or torch.Tensor): Single image to annotate.
        true_labels (torch.Tensor or list): True labels for the image.
        predictions (torch.Tensor or list): Predicted labels for the image.
        index_to_tag (dict): Mapping from label indices to tag names.

    Returns:
        PIL.Image: Annotated image.
    """
    # If the image is a tensor, convert it to a PIL Image first
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    # Get the size of the image
    width, height = image.size
    
    # Set the font size to be proportional to the width of the image
    font_size = int(width * 0.03)  # You can adjust the 0.03 factor as needed
    #font = ImageFont.truetype("arial.ttf", font_size)  # You can choose a different font if you like
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", font_size)

    draw = ImageDraw.Draw(image)

    pred_label_text = metricutils.convert_labels_to_string(predictions, index_to_tag)
    if true_labels is not None:
        true_label_text = metricutils.convert_labels_to_string(true_labels, index_to_tag)
        # Prepare text to be overlayed on the image
        text = f"True: {true_label_text}\nPred: {pred_label_text}"
    else:
        text = f"Pred: {pred_label_text}"

    # Set text position to be proportional to the size of the image
    text_x = width * 0.01  # You can adjust the 0.01 factor as needed
    text_y = height * 0.01  # You can adjust the 0.01 factor as needed

    # Draw the text on the image with the proportional font size
    draw.text((text_x, text_y), text, (57, 255, 20), font=font)  # Green text, top-left corner

    return image

def overlay_predictions_video(video_path, predictions,frame_counts, index_to_tag, output_path):
    # Open the input video
    video_capture = cv2.VideoCapture(str(video_path))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a VideoWriter object to save the annotated video
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize variables to track the most recent predictions
    last_prediction = None
    frame_idx = 0
    pred_idx = 0

    # Process the frames and overlay predictions
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        # Update last_prediction if the current frame is one of the predicted frames
        if pred_idx < len(frame_counts) and frame_idx == frame_counts[pred_idx]:
            last_prediction = predictions[pred_idx]
            pred_idx += 1

        # If there are predictions available, overlay them on the frame
        if last_prediction is not None:
            annotated_image = overlay_predictions(pil_image, last_prediction, index_to_tag)
            # Convert back to OpenCV image
            frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        video_writer.write(frame)
        frame_idx += 1

    # Release resources
    video_capture.release()
    video_writer.release()


def convert_labels_to_color(labels, num_classes, height=10, width=10):
    """
    Converts labels to a color representation using a colormap.

    Parameters:
        labels (torch.Tensor): Tensor of labels, either in class index form or one-hot encoded.
        num_classes (int): Number of classes.
        height (int): Height of the color image representation.
        width (int): Width of the color image representation.

    Returns:
        torch.Tensor: Color representation of labels in the shape [batch_size, channels, height, width].
    """
    # Generate a colormap
    cmap = plt.get_cmap('viridis', num_classes)  # Get the colormap

    # Convert labels to indices if they are one-hot encoded
    if labels.ndim > 1 and labels.size(1) == num_classes:
        labels = labels.argmax(dim=1)
    else:
        # If labels are not one-hot encoded, ensure they are integer class indices
        labels = labels.long()

    # Normalize label indices to be between 0 and 1
    labels_normalized = labels.float() / (num_classes - 1)

    # Map normalized indices to colors using the colormap
    colors = cmap(labels_normalized.numpy())[:, :3]  # Get the RGB values and exclude the alpha channel

    # Convert colors to a PyTorch tensor and reshape to [batch_size, 1, 1, channels]
    colors_tensor = torch.tensor(colors, dtype=torch.float32).view(-1, 1, 1, 3)

    # Repeat colors across the desired image dimensions to create a full image representation for each label
    colors_tensor = colors_tensor.repeat(1, height, width, 1)

    # Permute the tensor to match the [batch_size, channels, height, width] format
    colors_tensor = colors_tensor.permute(0, 3, 1, 2)

    return colors_tensor

def convert_PIL_to_tensors(pil_images):
    """
    Converts a list of PIL images to PyTorch tensors.

    Parameters:
        pil_images (list of PIL.Image): List of PIL images to convert.

    Returns:
        torch.Tensor: Batch of images as PyTorch tensors.
    """
    return torch.stack([transforms.ToTensor()(img) for img in pil_images])

def convert_to_PIL(images):
    """
    Converts a batch of PyTorch tensors to PIL images.

    Parameters:
        images (torch.Tensor or list of torch.Tensor): Batch of images to convert.

    Returns:
        list: List of PIL images.
    """
    to_pil = transforms.ToPILImage()
    return [to_pil(image) for image in images]

def preprocess_image(image_path, config):
    """Preprocess an image file to be suitable for model input.

     Parameters:
        image_path (str): Path of the image to process.
        config (object): Configuration desired image size and normalization parameters.
    
    """
    transforms = Compose([
        Resize(config.model_image_size),  # Resize to the input size expected by the model
        ToTensor(),          # Convert to PyTorch Tensor
        Normalize(config.dataset_normalization_mean, config.dataset_normalization_std) # Normalize with the same values used in training
    ])
    image = Image.open(image_path)
    return transforms(image).unsqueeze(0)  # Add batch dimension