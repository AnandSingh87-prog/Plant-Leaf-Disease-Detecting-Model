import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define image properties and directories
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/valid'
MODEL_DIR = 'models'

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Data Preprocessing and Augmentation
print("Preparing dataset generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values
    rotation_range=20,       # Randomly rotate images
    width_shift_range=0.2,   # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    horizontal_flip=True,    # Randomly flip images horizontally
    zoom_range=0.2           # Randomly zoom images
)

val_datagen = ImageDataGenerator(rescale=1./255) # Only rescale validation data

# Note: In a real scenario, make sure you put images in class folders inside dataset/train/ and dataset/validation/
try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
except FileNotFoundError:
    print("Warning: Dataset directories not found. Please place your class folders inside dataset/train/ and dataset/validation/")
    class MockGenerator:
        num_classes = 2 # Mocking a binary class if empty
    train_generator = MockGenerator()
    val_generator = MockGenerator()


# 2. Model Building (Transfer Learning with MobileNetV2)
print("Building the model...")
# Load the pre-trained MobileNetV2 model, excluding the top (classification) layers
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Freeze the layers of the base model so they aren't trained from scratch
base_model.trainable = False

# Add custom top layers for our specific prediction task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
num_classes = train_generator.num_classes if hasattr(train_generator, 'num_classes') else 2
predictions = Dense(num_classes, activation='softmax')(x) # Output layer

# Combine the base model and custom top layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# 3. Callbacks for Training
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'plant_disease_model.keras'), 
    monitor='val_accuracy', 
    save_best_only=True, 
    verbose=1
),

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# 4. Training (Uncomment and run when the dataset is populated)
print("-" * 50)
print("Model Built Successfully.")
print("To train the model, place your images in structured folders under dataset/train/ and dataset/validation/, then run this script.")
print("Example folder structure:")
print("dataset/")
print("  train/")
print("    Tomato___Healthy/")
print("      img1.jpg")
print("    Tomato___Late_blight/")
print("      img2.jpg")
print("-" * 50)

# The following lines are now uncommented since the dataset is populated.
model.fit(
    train_generator,
    epochs=5, # Reduced to 10 for a quicker test run
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping]
)

import json
# Save class indices for reference in the app
if hasattr(train_generator, 'class_indices'):
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
