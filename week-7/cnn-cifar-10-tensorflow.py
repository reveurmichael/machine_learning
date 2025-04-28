import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import numpy as np
import os

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load and preprocess CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between -1 and 1
train_images = (train_images / 127.5) - 1
test_images = (test_images / 127.5) - 1

# Model
def create_model():
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)
    ])
    
    return model

# Create and compile the model
model = create_model()
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Training parameters
BATCH_SIZE = 64
EPOCHS = 50

# Training
history = model.fit(
    train_images, train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(test_images, test_labels),
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Accuracy on the 10000 test images: {test_acc * 100:.2f}%")

# Save model
model_path = "model/tensorflow_cifar10.h5"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"Model saved to {model_path}")
