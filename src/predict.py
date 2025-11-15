# ============================================
# PREDICT.PY - Make Predictions on Test Images
# ============================================
# This script loads a trained model and makes predictions on test images.
# It supports configuration via YAML files for flexible usage.
#
# Usage:
#   python src/predict.py                    # Uses default config (predict.yaml)
#   python src/predict.py --config predict   # Uses configs/predict.yaml
#   python src/predict.py --config-path custom.yaml
# ============================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Setup paths and import shared utilities
from src.utils import setup_paths, get_project_root, ensure_dir, get_path, create_results_structure
setup_paths()

from src.config_loader import load_config, find_latest_model

PROJECT_ROOT = get_project_root()


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image for prediction.
    
    Parameters:
    ----------
    image_path : str
        Path to image file
    target_size : tuple
        Size to resize image to (must match training size)
    
    Returns:
    -------
    img_array : numpy array
        Preprocessed image ready for model (with batch dimension)
    """
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to target size
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to NumPy array and normalize
    img_array = np.array(img).astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_test_set(config_path=None, config_name=None):
    """
    Make predictions on all test images and save to CSV.
    
    Parameters:
    ----------
    config_path : str, optional
        Full path to configuration file
    config_name : str, optional
        Name of configuration file in configs/ directory
    
    Returns:
    -------
    predictions_df : pandas DataFrame
        DataFrame with predictions
    """
    # Load configuration
    config = load_config(config_path=config_path, config_name=config_name)
    
    # Extract configuration sections
    paths = config.get('paths', {})
    model_config = config.get('model', {})
    image_config = config.get('image', {})
    prediction_config = config.get('prediction', {})
    output_config = config.get('output', {})
    
    print("=" * 70)
    print("MAKING PREDICTIONS ON TEST SET")
    print("=" * 70)
    print()
    
    # ============================================
    # STEP 1: Load Trained Model
    # ============================================
    print("STEP 1: Loading trained model...")
    print("-" * 70)
    
    # Create results structure
    results_dir = create_results_structure()
    
    models_dir = get_path(paths, 'models_dir', os.path.join(PROJECT_ROOT, 'models'))
    model_path_config = model_config.get('path', 'latest')
    
    # Determine model path
    if model_path_config == 'latest':
        model_path = find_latest_model(models_dir, model_config.get('model_type'))
        if model_path is None:
            print(f"ERROR: No model files found in '{models_dir}'")
            print("Please train a model first using: python src/train.py")
            return None
    else:
        model_path = model_path_config
        if not os.path.isabs(model_path):
            model_path = os.path.join(models_dir, model_path)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None
    
    # Load the model
    model = keras.models.load_model(model_path)
    
    print(f"Model loaded from: {model_path}")
    print(f"  Input shape: {model.input_shape}")
    print()
    
    # ============================================
    # STEP 2: Get List of Test Images
    # ============================================
    print("STEP 2: Scanning test folder...")
    print("-" * 70)
    
    test_dir = get_path(paths, 'test_dir', os.path.join(PROJECT_ROOT, 'data', 'test'))
    
    # Get all image files
    image_files = [f for f in os.listdir(test_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sort by filename (assuming they're numbered)
    try:
        image_files.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)
    except:
        image_files.sort()
    
    print(f"Found {len(image_files)} test images")
    print()
    
    # ============================================
    # STEP 3: Make Predictions
    # ============================================
    print("STEP 3: Making predictions...")
    print("-" * 70)
    print("This may take a while...")
    print()
    
    # Get image size from config
    image_size = tuple(image_config.get('size', [224, 224]))
    batch_size = prediction_config.get('batch_size', 32)
    threshold = prediction_config.get('threshold', 0.5)
    
    # Store predictions
    predictions = []
    image_ids = []
    
    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        
        # Load and preprocess batch
        batch_images = []
        batch_ids = []
        
        for filename in batch_files:
            image_path = os.path.join(test_dir, filename)
            img_array = preprocess_image(image_path, image_size)
            batch_images.append(img_array[0])  # Remove batch dimension
            try:
                batch_ids.append(int(filename.split('.')[0]))
            except:
                batch_ids.append(i + len(batch_ids))
        
        # Convert to NumPy array
        batch_images = np.array(batch_images)
        
        # Make predictions
        batch_predictions = model.predict(batch_images, verbose=0)
        
        # Extract dog probabilities
        if len(batch_predictions.shape) > 1 and batch_predictions.shape[1] == 2:
            dog_probs = batch_predictions[:, 1]
        else:
            dog_probs = batch_predictions[:, 0] if len(batch_predictions.shape) > 1 else batch_predictions
        
        # Store predictions
        predictions.extend(dog_probs)
        image_ids.extend(batch_ids)
        
        # Print progress
        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(image_files):
            print(f"  Processed {min(i + batch_size, len(image_files))}/{len(image_files)} images...")
    
    print()
    print("Predictions complete")
    print()
    
    # ============================================
    # STEP 4: Save Predictions to CSV
    # ============================================
    print("STEP 4: Saving predictions to CSV...")
    print("-" * 70)
    
    # Create DataFrame - save to results/predictions
    output_csv = prediction_config.get('output_csv', 'results/predictions/predictions.csv')
    if not os.path.isabs(output_csv):
        output_csv = os.path.join(results_dir['predictions'], os.path.basename(output_csv))
    ensure_dir(os.path.dirname(output_csv))
    
    # Prepare data for DataFrame
    data_dict = {'id': image_ids}
    
    if output_config.get('include_confidence', True):
        data_dict['label'] = predictions
    else:
        # Binary predictions only
        data_dict['label'] = [1 if p > threshold else 0 for p in predictions]
    
    df = pd.DataFrame(data_dict)
    
    # Sort by ID if requested
    if output_config.get('sort_by_id', True):
        df = df.sort_values('id').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"Predictions saved to: {output_csv}")
    print(f"  Total predictions: {len(predictions)}")
    print()
    
    # ============================================
    # STEP 5: Show Statistics
    # ============================================
    print("STEP 5: Prediction statistics...")
    print("-" * 70)
    
    # Convert probabilities to binary predictions
    binary_predictions = (np.array(predictions) > threshold).astype(int)
    
    num_dogs = np.sum(binary_predictions == 1)
    num_cats = np.sum(binary_predictions == 0)
    
    print(f"  Predicted dogs: {num_dogs} ({num_dogs/len(predictions)*100:.1f}%)")
    print(f"  Predicted cats: {num_cats} ({num_cats/len(predictions)*100:.1f}%)")
    print(f"  Average dog probability: {np.mean(predictions):.3f}")
    print()
    
    # Show first few predictions
    print("Sample predictions (first 10):")
    print(df.head(10).to_string(index=False))
    print()
    
    print("=" * 70)
    print("PREDICTION COMPLETE!")
    print("=" * 70)
    print(f"Output file: {output_csv}")
    print()
    
    return df


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    """
    Main entry point for prediction script.
    Supports command-line arguments for configuration selection.
    """
    parser = argparse.ArgumentParser(
        description='Make predictions on test images using a trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/predict.py                    # Use default config (predict.yaml)
  python src/predict.py --config predict   # Use configs/predict.yaml
  python src/predict.py --config-path custom.yaml  # Use custom config file
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='predict',
        help='Name of config file in configs/ directory (without extension)'
    )
    
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Full path to custom configuration file'
    )
    
    args = parser.parse_args()
    
    # Make predictions
    predictions_df = predict_test_set(
        config_path=args.config_path,
        config_name=args.config if not args.config_path else None
    )
    
    if predictions_df is not None:
        print("Predictions completed successfully")
