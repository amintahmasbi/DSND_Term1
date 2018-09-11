import torch
from torchvision import datasets, transforms
import json
import numpy as np
import time

def load_data(data_dir):
    """
    Load the data using torchvision functions
    """

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Make function params from these variables
    scale_size = 256
    crop_size = 224
    max_rotation = 30
    batch_size = 32
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(max_rotation),
                                           transforms.RandomResizedCrop(crop_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(normalization_mean, 
                                                                normalization_std)])

    test_transforms = transforms.Compose([transforms.Resize(scale_size),
                                     transforms.CenterCrop(crop_size),
                                     transforms.ToTensor(),
                                         transforms.Normalize(normalization_mean, 
                                                                normalization_std)]) 

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, validloader, testloader, train_dataset.class_to_idx


def load_label_mapping(mapping_file):
    """
    Load in a mapping from category label to category name
    """
    with open(mapping_file, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    scale_size = 256
    crop_size = 224
    # Scale
    pil_image = image.resize((scale_size,scale_size))
    #Crop
    center_pixel = scale_size/2
    crop_top_left = center_pixel-(crop_size/2)
    crop_bottom_right = crop_top_left+crop_size
    pil_image = pil_image.crop((crop_top_left,crop_top_left,crop_bottom_right,crop_bottom_right))
    #Transfrom to [0-1] interval
    np_image = np.array(pil_image)/255
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    np_image = np_image.transpose((2, 0, 1))
    return np_image