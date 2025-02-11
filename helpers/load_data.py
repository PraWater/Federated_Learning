import logging
import os
import numpy as np
from PIL import Image, ImageEnhance
import random

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Image Parameters
IMG_SIZE = (224, 224)
DATA_DIR = "/app/dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")


# Manual Data Augmentation Function
def augment_image(img):
    """Apply augmentation using PIL."""
    # Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random rotation (-20 to +20 degrees)
    img = img.rotate(random.uniform(-20, 20))

    # Random brightness adjustment (0.8x to 1.2x)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    return img


# Function to Load and Process Images
def load_images_from_directory(directory, augment=False):
    """Load images and labels from a directory."""
    x_data, y_data = [], []
    class_names = sorted(os.listdir(directory))  # Ensure consistent class mapping
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")  # Ensure 3 channels
                    img = img.resize(IMG_SIZE)  # Resize to (224, 224)

                    if augment:
                        img = augment_image(img)  # Apply augmentation if training

                    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
                    x_data.append(img_array)
                    y_data.append(class_to_idx[class_name])  # Convert label to index
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(x_data), np.array(y_data)


# Main Data Loading Function
def load_data(data_sampling_percentage=0.5, client_id=1, total_clients=2):
    """Load dataset without TensorFlow, ensuring global shuffling and partitioning."""

    # Load images into numpy arrays
    x_train_full, y_train_full = load_images_from_directory(TRAIN_DIR, augment=True)
    x_test, y_test = load_images_from_directory(TEST_DIR, augment=False)

    # Shuffle globally before splitting
    np.random.seed(42)
    indices = np.arange(len(x_train_full))
    np.random.shuffle(indices)
    x_train_full, y_train_full = x_train_full[indices], y_train_full[indices]

    # Partitioning data among clients
    total_samples = len(x_train_full)
    samples_per_client = total_samples // total_clients
    start_idx = (client_id - 1) * samples_per_client
    end_idx = start_idx + samples_per_client

    x_train, y_train = x_train_full[start_idx:end_idx], y_train_full[start_idx:end_idx]

    # Apply client-specific sampling
    num_samples = int(data_sampling_percentage * len(x_train))
    sampled_indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train, y_train = x_train[sampled_indices], y_train[sampled_indices]

    return (x_train, y_train), (x_test, y_test)
