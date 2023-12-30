import os
import cv2
import torch
import utils.models.modelfactory as modelfactory
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset

# Define a simple Dataset just to load images from a folder
class FolderDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        # Add a filtering step to only include .png and .jpg files
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)
                            if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = read_image(image_path)  # Reads image into a tensor
        if self.transform:
            image = self.transform(image)
        return image, image_path


# Define the image transform
def get_transform():
    return transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.4980, 0.4057, 0.3608], std=[0.2668, 0.2423, 0.2294]),
    ])

# Function to perform inference and save predicted images
def run_inference(model, input_folder, output_folder, threshold=0.2):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize the dataset and dataloader
    dataset = FolderDataset(input_folder, transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Iterate over the dataloader
    for i, (image_tensor, image_path) in enumerate(dataloader):
        # Perform inference
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        outputs = torch.sigmoid(outputs).detach().cpu()

        # Apply threshold to get predictions
        predicted_indices = (outputs > threshold).nonzero(as_tuple=True)[1]

        # Get predicted tags
        predicted_values = [tags_dict[x.item()] for x in predicted_indices]

        # Load the original image
        original_image = cv2.imread(image_path[0])
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Annotate and save the image
        annotated_image_path = os.path.join(output_folder, f"prediction_{i}.jpg")
        cv2.putText(original_image, f"PREDICTED: {predicted_values}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(annotated_image_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

        print(f"Processed {image_path[0]} - Predicted tags: {predicted_values}")

def read_tags(file_path):
    with open(file_path, 'r') as file:
        tags_dict = {i: tag.strip() for i, tag in enumerate(file)}
    return tags_dict

# Define the main function
if __name__ == '__main__':
    # Initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 31  
    model = modelfactory.model(pretrained=False, requires_grad=False, numClasses=num_classes).to(device)
    # Load the model checkpoint
    checkpoint = torch.load('../outputs/best_model.pth')

    # Load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Define paths
    input_folder = '../inference_inputs'
    output_folder = '../inference_outputs'

    # Define your tags dictionary
    tags_dict = read_tags("D:/test/NeuralDataSet/SexualActionsDataset/tags.txt")

    # Run inference
    run_inference(model, input_folder, output_folder)
