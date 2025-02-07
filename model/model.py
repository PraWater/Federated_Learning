import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import ResNet50


class Model:
    def __init__(self, learning_rate, num_classes):
        self.learning_rate = learning_rate
        self.loss_function = tf.keras.losses.CategoricalCrossentropy()  # Updated for categorical labels

        # Load ResNet50 with pretrained weights
        self.base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        self.base_model.trainable = False  # Freeze the base model

        # Build model with additional layers
        self.model = models.Sequential([
            self.base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax")
        ])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compile(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=["accuracy"]
        )

    def get_model(self):
        return self.model
