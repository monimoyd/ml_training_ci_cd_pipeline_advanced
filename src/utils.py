import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torchvision import datasets, transforms


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, weights_only=False)
    model.to(device)
    model.eval()
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return correct / total
    
def add_gaussian_noise(image, mean=0., std=0.1):
    noise = torch.randn(image.size()) * std + mean
    return torch.clamp(image + noise, 0, 1)

def random_rotation(image, max_angle=30):
    angle = torch.randint(-max_angle, max_angle, (1,)).item()
    return transforms.functional.rotate(image, angle)

def random_brightness(image, factor_range=(0.5, 1.5)):
    factor = torch.empty(1).uniform_(factor_range[0], factor_range[1]).item()
    return torch.clamp(image * factor, 0, 1)
    
def visualize_mnist_augmentations():
    """
    Visualize augmentations on one example of each MNIST digit (0-9)
    """
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Get one example of each digit
    digit_examples = {}
    for img, label in mnist_data:
        if label not in digit_examples:
            digit_examples[label] = img
        if len(digit_examples) == 10:
            break
    
    # Create figure
    fig, axes = plt.subplots(10, 4, figsize=(12, 25))
    plt.subplots_adjust(hspace=0.3)
    
    # For each digit
    for digit in range(10):
        img = digit_examples[digit]
        
        # Create augmented versions
        noisy_img = add_gaussian_noise(img.clone())
        rotated_img = random_rotation(img.clone())
        bright_img = random_brightness(img.clone())
        
        # Plot original and augmented versions
        images = [img, noisy_img, rotated_img, bright_img]
        titles = ['Original', 'Noisy', 'Rotated', 'Brightness']
        
        for col, (image, title) in enumerate(zip(images, titles)):
            ax = axes[digit, col]
            if col == 0:  # Add digit label on the left
                ax.set_ylabel(f'Digit {digit}', rotation=0, labelpad=40)
            if digit == 0:  # Add augmentation type on top
                ax.set_title(title)
            ax.imshow(image.squeeze(), cmap='gray')
            ax.axis('off')
    
    plt.suptitle('MNIST Digits with Different Augmentations', fontsize=16, y=1.02)
    #plt.show()
    plt.savefig('all_digit_augmentation.png')

def visualize_augmentation_sequence(digit_idx=5, num_sequences=5):
    """
    Visualize a sequence of augmentations applied to the same digit
    Args:
        digit_idx: Index of the digit to visualize (0-9)
        num_sequences: Number of different augmentation sequences to show
    """
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Get example of specified digit
    digit_example = None
    for img, label in mnist_data:
        if label == digit_idx:
            digit_example = img
            break
    
    # Create figure
    fig, axes = plt.subplots(num_sequences, 4, figsize=(12, 2.5*num_sequences))
    plt.subplots_adjust(hspace=0.3)
    
    # For each sequence
    for seq in range(num_sequences):
        img = digit_example.clone()
        
        # Create augmented versions
        noisy_img = add_gaussian_noise(img.clone(), std=0.1*(seq+1))
        rotated_img = random_rotation(img.clone(), max_angle=15*(seq+1))
        bright_img = random_brightness(img.clone(), 
                                    factor_range=(0.5-0.1*seq, 1.5+0.1*seq))
        
        images = [img, noisy_img, rotated_img, bright_img]
        titles = ['Original', 'Noisy', 'Rotated', 'Brightness']
        
        for col, (image, title) in enumerate(zip(images, titles)):
            ax = axes[seq, col]
            if seq == 0:
                ax.set_title(title)
            ax.imshow(image.squeeze(), cmap='gray')
            ax.axis('off')
    
    plt.suptitle(f'Multiple Augmentation Sequences for Digit {digit_idx}', 
                fontsize=14, y=1.02)
    #plt.show()
    plt.savefig('multiple_augmentation_same_digit.png')



if __name__ == '__main__':
    # Visualize augmentations
    print("Generating visualization of all digits...")
    visualize_mnist_augmentations()
    
    print("\nGenerating visualization of augmentation sequences...")
    visualize_augmentation_sequence()    
    
