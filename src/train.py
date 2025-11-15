# ============================================
# TRAIN.PY - Train the Neural Network Model
# ============================================
# This script trains a neural network model for image classification.
# It supports configuration via YAML files for flexible experimentation.
#
# Usage:
#   python src/train.py                    # Uses default config (train.yaml)
#   python src/train.py --config train     # Uses configs/train.yaml
#   python src/train.py --config debug     # Uses configs/debug.yaml
#   python src/train.py --config-path path/to/custom.yaml
# ============================================

import os
import sys
import argparse
import shutil
import numpy as np
from datetime import datetime

# Import TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras

# Setup paths and import shared utilities
from src.utils import setup_paths, get_project_root, ensure_dir, get_path, create_results_structure
setup_paths()

# Import custom modules
from src.config_loader import load_config
from src.data_loader import load_images_from_folder, split_data
from src.model import (
    create_simple_cnn_model,
    create_advanced_cnn_model,
    create_deep_custom_cnn_model,
    create_transfer_learning_model
)

PROJECT_ROOT = get_project_root()


def train_model(config_path=None, config_name=None):
    """
    Main training function that orchestrates the entire training process.
    
    Parameters:
    ----------
    config_path : str, optional
        Full path to configuration file
    config_name : str, optional
        Name of configuration file in configs/ directory (without extension)
    
    Returns:
    -------
    model : Keras Model
        Trained model
    history : History
        Training history object
    """
    # Load configuration
    config = load_config(config_path=config_path, config_name=config_name)
    
    # Extract configuration sections
    paths = config.get('paths', {})
    image_config = config.get('image', {})
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    callbacks_config = config.get('callbacks', {})
    output_config = config.get('output', {})
    logging_config = config.get('logging', {})
    
    # Set random seeds for reproducibility
    random_seed = data_config.get('random_seed', 42)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    print("=" * 70)
    print("DOGS VS CATS CLASSIFICATION - MODEL TRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {config_name or config_path or 'default'}")
    print()
    
    # ============================================
    # STEP 1: Load Training Data
    # ============================================
    print("STEP 1: Loading training data...")
    print("-" * 70)
    
    # Get paths using shared utility
    train_dir = get_path(paths, 'train_dir', os.path.join(PROJECT_ROOT, 'data', 'train'))
    image_size = tuple(image_config.get('size', [224, 224]))
    max_images = data_config.get('max_training_images')
    
    # Create results structure
    results_dir = create_results_structure()
    
    # Load images
    images, labels, filenames = load_images_from_folder(
        folder_path=train_dir,
        target_size=image_size,
        max_images=max_images
    )
    
    if len(images) == 0:
        print(f"ERROR: No images loaded! Check the '{train_dir}' folder path.")
        return None, None
    
    print(f"Loaded {len(images)} images")
    print(f"  - Cats: {np.sum(labels == 0)}")
    print(f"  - Dogs: {np.sum(labels == 1)}")
    print()
    
    # ============================================
    # STEP 2: Split Data into Train/Validation
    # ============================================
    print("STEP 2: Splitting data into training and validation sets...")
    print("-" * 70)
    
    validation_split = data_config.get('validation_split', 0.2)
    X_train, X_val, y_train, y_val = split_data(
        images, 
        labels, 
        validation_split=validation_split,
        random_seed=random_seed
    )
    
    print("Data split complete")
    print()
    
    # ============================================
    # STEP 3: Create Model
    # ============================================
    print("STEP 3: Creating neural network model...")
    print("-" * 70)
    
    # Determine input shape
    input_shape = tuple(image_config.get('input_shape', [*image_size, image_config.get('channels', 3)]))
    if len(input_shape) == 2:
        input_shape = (*input_shape, image_config.get('channels', 3))
    
    # Get model type from configuration
    # Available types: simple_cnn, advanced_cnn, deep_custom_cnn, transfer_learning
    # Each type represents a different architecture with varying complexity and accuracy
    model_type = model_config.get('type', 'simple_cnn')
    
    # Create model based on type specified in configuration
    # Each model type has different architecture, complexity, and training characteristics
    if model_type == 'simple_cnn':
        # Simple CNN: 3 convolutional blocks, moderate complexity, trains from scratch
        # Best for: Learning basics, moderate accuracy (~75-85%), faster training
        model = create_simple_cnn_model(
            input_shape=input_shape,
            num_classes=model_config.get('num_classes', 2),
            config=model_config.get('simple_cnn', {})
        )
    elif model_type == 'advanced_cnn':
        # Advanced CNN: 4 convolutional blocks, higher complexity, trains from scratch
        # Best for: Better accuracy (~80-88%), requires more training time
        model = create_advanced_cnn_model(
            input_shape=input_shape,
            num_classes=model_config.get('num_classes', 2),
            config=model_config.get('advanced_cnn', {})
        )
    elif model_type == 'deep_custom_cnn':
        # Deep Custom CNN: 5+ convolutional blocks, maximum depth, trains from scratch
        # Best for: High accuracy potential (~82-90%), significant training time
        model = create_deep_custom_cnn_model(
            input_shape=input_shape,
            num_classes=model_config.get('num_classes', 2),
            config=model_config.get('deep_custom_cnn', {})
        )
    elif model_type == 'transfer_learning':
        # Transfer Learning: Pre-trained base model with custom layers, fine-tuned
        # Best for: Best accuracy (~85-92%), fastest training, industry standard
        model = create_transfer_learning_model(
            input_shape=input_shape,
            num_classes=model_config.get('num_classes', 2),
            config=model_config.get('transfer_learning', {})
        )
    else:
        # Invalid model type specified in configuration
        # Display error message with available options
        print(f"ERROR: Unknown model type '{model_type}'")
        print("Available options: 'simple_cnn', 'advanced_cnn', 'deep_custom_cnn', 'transfer_learning'")
        return None, None
    
    print("Model created")
    print()
    
    # ============================================
    # STEP 4: Set Up Callbacks
    # ============================================
    print("STEP 4: Setting up training callbacks...")
    print("-" * 70)
    
    # Create directories using shared utilities
    logs_dir = get_path(paths, 'logs_dir', os.path.join(PROJECT_ROOT, 'logs'))
    models_dir = get_path(paths, 'models_dir', os.path.join(PROJECT_ROOT, 'models'))
    ensure_dir(models_dir)
    ensure_dir(logs_dir)
    
    # Generate model filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if output_config.get('include_timestamp', True) else ''
    prefix = output_config.get('model_name_prefix', 'dogs_vs_cats')
    model_filename = os.path.join(models_dir, f'{prefix}_{model_type}_{timestamp}.h5')
    
    # Setup callbacks
    callbacks = []
    
    # ModelCheckpoint
    checkpoint_cfg = callbacks_config.get('checkpoint', {})
    if checkpoint_cfg.get('enabled', True):
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=model_filename,
            monitor=checkpoint_cfg.get('monitor', 'val_accuracy'),
            save_best_only=checkpoint_cfg.get('save_best_only', True),
            mode=checkpoint_cfg.get('mode', 'max'),
            verbose=checkpoint_cfg.get('verbose', 1)
        ))
    
    # EarlyStopping
    early_stop_cfg = callbacks_config.get('early_stopping', {})
    if early_stop_cfg.get('enabled', True):
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=early_stop_cfg.get('monitor', 'val_accuracy'),
            patience=early_stop_cfg.get('patience', 5),
            restore_best_weights=early_stop_cfg.get('restore_best_weights', True),
            verbose=early_stop_cfg.get('verbose', 1)
        ))
    
    # ReduceLROnPlateau
    reduce_lr_cfg = callbacks_config.get('reduce_lr', {})
    if reduce_lr_cfg.get('enabled', True):
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor=reduce_lr_cfg.get('monitor', 'val_loss'),
            factor=reduce_lr_cfg.get('factor', 0.5),
            patience=reduce_lr_cfg.get('patience', 3),
            min_lr=reduce_lr_cfg.get('min_lr', 0.00001),
            verbose=reduce_lr_cfg.get('verbose', 1)
        ))
    
    print("Callbacks configured")
    if callbacks:
        print(f"  - Model will be saved to: {model_filename}")
    print()
    
    # ============================================
    # STEP 5: Train the Model
    # ============================================
    print("STEP 5: Training the model...")
    print("-" * 70)
    
    batch_size = training_config.get('batch_size', 32)
    epochs = training_config.get('epochs', 10)
    verbose = training_config.get('verbose', 1)
    
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print()
    print("Training in progress...")
    print()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose
    )
    
    print()
    print("Training complete")
    print()
    
    # ============================================
    # STEP 6: Evaluate Final Performance
    # ============================================
    print("STEP 6: Evaluating model performance...")
    print("-" * 70)
    
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")
    print()
    
    # ============================================
    # STEP 7: Save Final Model
    # ============================================
    if output_config.get('save_final_model', True):
        print("STEP 7: Saving final model...")
        print("-" * 70)
        
        # Save model to models directory
        # Note: Models should be manually organized into models/partial_dataset/ or models/full_dataset/
        # based on the dataset used for training (check data.max_training_images in config)
        final_model_filename = os.path.join(models_dir, f'{prefix}_{model_type}_final_{timestamp}.h5')
        model.save(final_model_filename)
        
        # Also copy to results/models for presentation
        results_model_path = os.path.join(results_dir['models'], os.path.basename(final_model_filename))
        shutil.copy2(final_model_filename, results_model_path)
        
        print(f"Model saved to: {final_model_filename}")
        print(f"Model copied to: {results_model_path}")
        print("Note: Organize models into models/partial_dataset/ or models/full_dataset/ for comparison")
        print()
    
    # ============================================
    # SUMMARY
    # ============================================
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Model type: {model_type}")
    print(f"Training images: {len(X_train)}")
    print(f"Validation images: {len(X_val)}")
    print(f"Final accuracy: {val_accuracy * 100:.2f}%")
    if output_config.get('save_final_model', True):
        print(f"Model saved: {final_model_filename}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return model, history


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    """
    Main entry point for training script.
    Supports command-line arguments for configuration selection.
    """
    parser = argparse.ArgumentParser(
        description='Train a neural network model for Dogs vs Cats classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/train.py                    # Use default config (train.yaml)
  python src/train.py --config train     # Use configs/train.yaml
  python src/train.py --config debug     # Use configs/debug.yaml
  python src/train.py --config-path custom.yaml  # Use custom config file
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='train',
        help='Name of config file in configs/ directory (without extension)'
    )
    
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Full path to custom configuration file'
    )
    
    args = parser.parse_args()
    
    # Train the model
    model, history = train_model(
        config_path=args.config_path,
        config_name=args.config if not args.config_path else None
    )
    
    if model is not None:
        print("\nTraining completed successfully")
        print("The saved model can now be used to make predictions.")
