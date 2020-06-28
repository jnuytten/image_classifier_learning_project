import json
import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image


def create_dataloader(data_directory, batch_size):
    """
    Loads images for training, validation and testing.
    Executes transforms and normalize images.
    
    Parameters
    ----------
    data_directory : string containing path to data root
    batch_size: size of dataloader batches
        

    Returns
    -------
    dataloader : list containing the trainloader, validationloader and testloader.

    """
    print(f"Loading data from {data_directory}")
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'
    
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    valtest_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valtest_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = valtest_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    # Compose and return dataset and dataloader
    dataset = [train_data, valid_data, test_data]
    dataloader = [trainloader, validloader, testloader]
        
    return dataset, dataloader
    

def get_labels(category_names):
    """
    Creates dictionary mapping labels to category names from JSON file.
    Parameters
    ----------
    category_names : string containing path to JSON file

    Returns
    -------
    cat_to_name : dictionary mapping labels to category names

    """
    print(f"Loading category names from {category_names}")
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    
def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model

    Parameters
    ----------
    image_path : string containing path to image file

    Returns
    -------
    numpy array

    """
    print(f"Processing image {image_path}")
    im = Image.open(image_path)
    # Resize keeping aspect ratio
    im.thumbnail((256, 256))
    # Center crop 224x224
    left_up = 0.5*(256-224)
    im = im.crop((left_up, left_up, 256 - left_up, 256 - left_up))
    # Normalize
    np_image = np.array(im) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    # Change color channel to be the first dimenstion and return np array
    return np_image.transpose((2, 0, 1)) 