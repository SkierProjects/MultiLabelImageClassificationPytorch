from src.config import config
import torchvision.transforms as transforms
import torch
from PIL import ImageDraw
import matplotlib.pyplot as plt

def denormalize_images(images, config=config):
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

def overlay_predictions(images, true_labels, predictions, index_to_tag):
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
        # If the image is a tensor, convert it to a PIL Image first
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
            
        draw = ImageDraw.Draw(img)
        # Convert tensors to lists if they are not already
        true_label_list = true_label_vec.tolist() if isinstance(true_label_vec, torch.Tensor) else true_label_vec
        pred_label_list = pred_label_vec.tolist() if isinstance(pred_label_vec, torch.Tensor) else pred_label_vec

        # Generate the label text using the index_to_tag mapping
        true_label_text = ','.join(index_to_tag[i] for i, label in enumerate(true_label_list) if label == 1)
        pred_label_text = ','.join(index_to_tag[i] for i, label in enumerate(pred_label_list) if label == 1)

        # Prepare text to be overlayed on the image
        text = f"True: {true_label_text}\nPred: {pred_label_text}"
        
        draw.text((0, 0), text, (57, 255, 20))  # Green text, top-left corner
        annotated_images.append(img)

    return annotated_images

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