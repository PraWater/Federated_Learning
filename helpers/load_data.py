import logging

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


def load_data(data_sampling_percentage=0.5, client_id=1, total_clients=2):
    """Load MRI dataset, ensure global shuffling, and return client-specific NumPy arrays.

    Args:
        data_sampling_percentage (float): Percentage of the dataset to use for training.
        client_id (int): Unique ID for the client.
        total_clients (int): Total number of clients.

    Returns:
        Tuple of arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    train_dir = "/app/dataset/Training"
    test_dir = "/app/dataset/Testing"

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False  # No shuffle here to ensure consistent partitioning
    )

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    def generator_to_numpy(generator):
        x_data = []
        y_data = []
        for _ in range(len(generator)):
            x_batch, y_batch = generator.next()
            x_data.append(x_batch)
            y_data.append(y_batch)
        return np.concatenate(x_data, axis=0), np.concatenate(y_data, axis=0)

    x_train_full, y_train_full = generator_to_numpy(train_generator)
    x_test, y_test = generator_to_numpy(test_generator)

    np.random.seed(42)  # Ensures reproducibility
    indices = np.arange(len(x_train_full))
    np.random.shuffle(indices)
    x_train_full, y_train_full = x_train_full[indices], y_train_full[indices]

    total_samples = len(x_train_full)
    samples_per_client = total_samples // total_clients
    start_idx = (client_id - 1) * samples_per_client
    end_idx = start_idx + samples_per_client

    x_train, y_train = x_train_full[start_idx:end_idx], y_train_full[start_idx:end_idx]

    num_samples = int(data_sampling_percentage * len(x_train))
    sampled_indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train, y_train = x_train[sampled_indices], y_train[sampled_indices]

    return (x_train, y_train), (x_test, y_test)
