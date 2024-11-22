import unittest
import sys
import numpy as np
import torch
import src.model
from src.model import MNISTModel
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from src.utils import count_parameters, validate_model, add_gaussian_noise, random_rotation, random_brightness
sys.path.append('../src')

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = MNISTModel()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.train_dataset = MNIST(root='./data', train=True, 
                                 download=True, transform=self.transform)
        self.test_dataset = MNIST(root='./data', train=False, 
                                download=True, transform=self.transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=1000, 
                                     shuffle=False)
    
    def test_parameter_count(self):
        param_count = count_parameters(self.model)
        self.assertLess(param_count, 25000, "Model has too many parameters")
    
    def test_input_shape(self):
        test_input = torch.randn(1, 1, 28, 28)
        try:
            output = self.model(test_input)
            self.assertEqual(output.shape[1], 10, "Output shape is incorrect")
        except Exception as e:
            self.fail(f"Model failed to process 28x28 input: {str(e)}")
    
    def test_model_accuracy(self):
        accuracy = validate_model('latest_model.pth')
        self.assertGreater(accuracy, 0.95, "Model accuracy is below 95%")
        
    def test_rotation_invariance(self):
        test_input = torch.rand(1, 1, 28, 28)
        rotated_input = random_rotation(test_input)
        outputs = []
        for _ in range(5):  # Test multiple rotations
            output = self.model(rotated_input)
            outputs.append(torch.softmax(output, dim=1))
        
        # Check if probability distributions are similar across rotations
        max_diff = torch.max(torch.abs(outputs[0] - outputs[-1]))
        self.assertLess(max_diff, 0.3, "Model should be somewhat rotation invariant")

    
    def test_dataset_size_and_shape(self):
        """Test the size and shape of MNIST dataset"""
        # Check dataset sizes
        self.assertEqual(len(self.train_dataset), 60000, 
                        "Training set should contain 60000 images")
        self.assertEqual(len(self.test_dataset), 10000, 
                        "Test set should contain 10000 images")
        
        # Check image dimensions
        train_image, _ = self.train_dataset[0]
        self.assertEqual(train_image.shape, (1, 28, 28), 
                        "Images should be 1x28x28")
        
        # Check data type
        self.assertEqual(train_image.dtype, torch.float32, 
                        "Images should be float32")
        
        # Check value range
        self.assertTrue(torch.all((train_image >= 0) & (train_image <= 1)), 
                       "Pixel values should be normalized between 0 and 1")

    def test_class_distribution(self):
        """Test if classes are balanced in the dataset"""
        # Count occurrences of each digit
        labels = []
        for _, label in self.train_loader:
            labels.extend(label.numpy())
        
        label_counts = np.bincount(labels)
        
        # Check if each digit appears between 5400 and 6400 times (roughly balanced)
        for digit in range(10):
            count = label_counts[digit]
            self.assertTrue(5400 <= count <= 7000, 
                          f"Digit {digit} has {count} samples, "
                          f"expected between 5400 and 6400")
        
        # Check if we have exactly 10 classes
        self.assertEqual(len(np.unique(labels)), 10, 
                        "Dataset should have exactly 10 classes")

    def test_data_integrity(self):
        """Test data integrity and statistical properties"""
        # Get a batch of images
        images, _ = next(iter(self.train_loader))
        
        # Check for NaN or infinity values
        self.assertFalse(torch.isnan(images).any(), 
                        "Dataset contains NaN values")
        self.assertFalse(torch.isinf(images).any(), 
                        "Dataset contains infinite values")
        
        # Check basic statistical properties
        batch_mean = images.mean().item()
        batch_std = images.std().item()
        
        # MNIST typically has these statistical properties
        self.assertTrue(0.1 <= batch_mean <= 0.2, 
                       f"Unusual mean value: {batch_mean:.3f}")
        self.assertTrue(0.2 <= batch_std <= 0.35, 
                       f"Unusual standard deviation: {batch_std:.3f}")
        
        # Check for dead pixels (pixels that are always 0 or 1)
        batch_min = images.min(dim=0)[0]
        batch_max = images.max(dim=0)[0]
        
        dead_pixels = ((batch_min == batch_max).sum().item())
        total_pixels = 28 * 28
        dead_pixel_ratio = dead_pixels / total_pixels
        
        self.assertLess(dead_pixel_ratio, 0.25, 
                       f"Too many dead pixels: {dead_pixel_ratio:.2%}")

if __name__ == '__main__':
    unittest.main()