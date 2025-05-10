""" This Fine tuned model is trained only on Apples and Bananas from the linked dataset.
The images in the dataset are not very ideal for this project.
If you wish to experiment you can create a dataset of your own.
However if you use the linked dataset from the Readme, I would suggest you place the fruits in a white background while testing, or optimize the model on your own for it to perform better. """

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define paths
DATASET_DIR = r"insert/your/path/to/dataset/here"  # Path to your dataset
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20 # Can be increased for better performance

# Data augmentation for training (non-ideal conditions)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Increased rotation
    shear_range=0.2,    # Shear transformations
    zoom_range=0.2,     # Random zoom
    horizontal_flip=True,  # Horizontal flips
    brightness_range=[0.7, 1.3],  # Random brightness variation
    channel_shift_range=50.0,  # Randomly change brightness/color
    fill_mode="nearest",
    validation_split=0.2  
)

# Test data generator 
test_datagen = ImageDataGenerator(
    rescale=1./255  # Only rescale for test data
)

# Training data generator (with data augmentation)
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "Train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"  # Using training subset (80% of Train folder)
)

# Validation data generator (from Train folder or you can use Test folder)
val_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "Train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"  # Using validation subset (20% of Train folder)
)

# Test data generator (using the separate Test folder)
test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "Test"),  # Test folder
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Load pre-trained MobileNetV2 model (without top classification layers)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze all layers of the base model initially
base_model.trainable = False

# Add custom classification head on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling
x = Dense(128, activation='relu')(x)  # Dense layer
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Output layer

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Unfreeze the last few layers of MobileNetV2 for fine-tuning
for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers 
    layer.trainable = True

# Re-compile the model after unfreezing
model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

# Fine-tune the model
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Save the model in SavedModel format
saved_model_dir = "mobilenet_fresh_stale_finetuned_saved_model"
model.save(saved_model_dir, save_format="tf")
print(f"Fine-tuned model saved in SavedModel format at {saved_model_dir}")
