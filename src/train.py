# ============================================
# TRAIN.PY - Train the Neural Network Model
# ============================================
# This script:
#   1. Loads and preprocesses the training data
#   2. Creates the neural network model
#   3. Trains the model on the data
#   4. Saves the trained model
#   5. Evaluates performance
# ============================================

# Import necessary libraries
import os
import numpy as np
from datetime import datetime

# Import TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras

# Import our custom modules
# These are the files we created in the src/ folder
import sys
# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add parent directory to path for imports
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_images_from_folder, split_data
from src.model import create_simple_model, create_advanced_model, create_transfer_learning_model

# ============================================
# CONFIGURATION (Settings)
# ============================================
# These are parameters you can adjust

# Image settings
IMAGE_SIZE = (224, 224)  # Size to resize images to (width, height)
# Common sizes: (224, 224), (128, 128), (256, 256)
# Larger = better accuracy but slower training

# Training settings
BATCH_SIZE = 32  # Number of images processed at once
# Larger batch = faster but uses more memory
# Common values: 16, 32, 64, 128

EPOCHS = 10  # Number of times to go through entire dataset
# 10 epochs is good for full training - gives model time to learn
# More epochs = more training, but risk of overfitting

VALIDATION_SPLIT = 0.2  # 20% of data for validation
# Validation set: Used to check model performance during training
# Not used for training, only for evaluation

# Model settings
MODEL_TYPE = 'transfer'  # Options: 'simple', 'advanced', 'transfer'
# 'simple': Basic CNN (fast, good for learning)
# 'advanced': More layers (slower, better accuracy)
# 'transfer': Transfer learning (best accuracy, requires internet for first run)

# Data settings
MAX_TRAINING_IMAGES = 5000  # Using 5000 images for balanced training (good for presentation)
# Change to None for full training (all 25K images), or set number like 1000, 5000 for testing
# For presentation comparison: 5000 images gives good results in reasonable time

# ============================================
# FUNCTION: Train the Model
# ============================================
def train_model():
    """
    Main training function.
    This orchestrates the entire training process.
    """
    
    print("=" * 70)
    print("DOGS VS CATS CLASSIFICATION - MODEL TRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ============================================
    # STEP 1: Load Training Data
    # ============================================
    print("STEP 1: Loading training data...")
    print("-" * 70)
    
    # Load images from train folder
    # This function is defined in data_loader.py
    data_train_path = os.path.join(PROJECT_ROOT, 'data', 'train')
    images, labels, filenames = load_images_from_folder(
        folder_path=data_train_path,
        target_size=IMAGE_SIZE,
        max_images=MAX_TRAINING_IMAGES
    )
    
    # Check if we have data
    if len(images) == 0:
        print(f"ERROR: No images loaded! Check the '{data_train_path}' folder path.")
        return
    
    print(f"✓ Loaded {len(images)} images")
    print(f"  - Cats: {np.sum(labels == 0)}")
    print(f"  - Dogs: {np.sum(labels == 1)}")
    print()
    
    # ============================================
    # STEP 2: Split Data into Train/Validation
    # ============================================
    print("STEP 2: Splitting data into training and validation sets...")
    print("-" * 70)
    
    # Split data: 80% for training, 20% for validation
    X_train, X_val, y_train, y_val = split_data(
        images, 
        labels, 
        validation_split=VALIDATION_SPLIT
    )
    
    print("✓ Data split complete")
    print()
    
    # ============================================
    # STEP 3: Create Model
    # ============================================
    print("STEP 3: Creating neural network model...")
    print("-" * 70)
    
    # Choose model type based on configuration
    if MODEL_TYPE == 'simple':
        model = create_simple_model(input_shape=(*IMAGE_SIZE, 3))
    elif MODEL_TYPE == 'advanced':
        model = create_advanced_model(input_shape=(*IMAGE_SIZE, 3))
    elif MODEL_TYPE == 'transfer':
        model = create_transfer_learning_model(input_shape=(*IMAGE_SIZE, 3))
    else:
        print(f"ERROR: Unknown model type '{MODEL_TYPE}'")
        return
    
    print("✓ Model created")
    print()
    
    # ============================================
    # STEP 4: Set Up Callbacks (Optional but Useful)
    # ============================================
    print("STEP 4: Setting up training callbacks...")
    print("-" * 70)
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = os.path.join(models_dir, f'dogs_vs_cats_{MODEL_TYPE}_{timestamp}.h5')
    
    # Callbacks: Functions called during training
    
    # ModelCheckpoint: Save model after each epoch if it improves
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=model_filename,
        monitor='val_accuracy',  # Watch validation accuracy
        save_best_only=True,      # Only save if better than previous
        mode='max',               # Maximize accuracy
        verbose=1                 # Print messages
    )
    
    # EarlyStopping: Stop training if model stops improving
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',   # Watch validation accuracy
        patience=5,               # Wait 5 epochs without improvement
        restore_best_weights=True, # Use best weights when stopping
        verbose=1
    )
    
    # ReduceLROnPlateau: Reduce learning rate if stuck
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',       # Watch validation loss
        factor=0.5,               # Reduce learning rate by half
        patience=3,               # Wait 3 epochs
        min_lr=0.00001,           # Minimum learning rate
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    print("✓ Callbacks configured")
    print(f"  - Model will be saved to: {model_filename}")
    print()
    
    # ============================================
    # STEP 5: Train the Model
    # ============================================
    print("STEP 5: Training the model...")
    print("-" * 70)
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print()
    print("This may take a while... Please be patient!")
    print()
    
    # Train the model
    # This is where the magic happens!
    # The model learns to distinguish dogs from cats
    history = model.fit(
        X_train, y_train,                    # Training data and labels
        batch_size=BATCH_SIZE,               # Images per batch
        epochs=EPOCHS,                       # Number of training rounds
        validation_data=(X_val, y_val),      # Validation data
        callbacks=callbacks,                  # Callbacks (save, early stop, etc.)
        verbose=1                            # Print progress (1 = detailed)
    )
    
    print()
    print("✓ Training complete!")
    print()
    
    # ============================================
    # STEP 6: Evaluate Final Performance
    # ============================================
    print("STEP 6: Evaluating model performance...")
    print("-" * 70)
    
    # Evaluate on validation set
    # This tells us how well the model performs
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")
    print()
    
    # ============================================
    # STEP 7: Save Final Model
    # ============================================
    print("STEP 7: Saving final model...")
    print("-" * 70)
    
    # Save the final model (even if not the best)
    final_model_filename = os.path.join(models_dir, f'dogs_vs_cats_{MODEL_TYPE}_final_{timestamp}.h5')
    model.save(final_model_filename)
    
    print(f"✓ Model saved to: {final_model_filename}")
    print()
    
    # ============================================
    # SUMMARY
    # ============================================
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Model type: {MODEL_TYPE}")
    print(f"Training images: {len(X_train)}")
    print(f"Validation images: {len(X_val)}")
    print(f"Final accuracy: {val_accuracy * 100:.2f}%")
    print(f"Model saved: {final_model_filename}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return model, history


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    """
    Run this script to train the model:
    python src/train.py
    """
    
    # Set random seeds for reproducibility
    # This ensures results are consistent across runs
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train the model
    model, history = train_model()
    
    print("\n🎉 Training completed successfully!")
    print("You can now use the saved model to make predictions.")

