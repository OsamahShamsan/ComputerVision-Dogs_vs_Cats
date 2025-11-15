# ============================================
# PREDICT.PY - Make Predictions on Test Images
# ============================================
# This script:
#   1. Loads a trained model
#   2. Loads test images
#   3. Makes predictions (dog or cat)
#   4. Saves predictions to CSV file (for submission)
# ============================================

# Import necessary libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Get project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'dogs_vs_cats_transfer_final_20251115_154323.h5')  # Path to your trained model
# Using Transfer Learning model (best accuracy ~97%+)
# This is the most recent and best performing model

TEST_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'test')  # Folder containing test images
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'predictions.csv')  # Output file for predictions
IMAGE_SIZE = (224, 224)  # Must match training image size

# ============================================
# FUNCTION: Load and Preprocess Single Image
# ============================================
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
        Preprocessed image ready for model
    """
    
    # Open image
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to target size
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to NumPy array
    img_array = np.array(img)
    
    # Normalize to [0, 1] range
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    # Model expects shape (batch_size, height, width, channels)
    # Single image needs shape (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


# ============================================
# FUNCTION: Make Predictions on Test Set
# ============================================
def predict_test_set(model_path, test_folder, output_csv, batch_size=32):
    """
    Make predictions on all test images and save to CSV.
    
    Parameters:
    ----------
    model_path : str
        Path to trained model file (.h5)
    test_folder : str
        Folder containing test images
    output_csv : str
        Output CSV file path
    batch_size : int
        Number of images to process at once
    """
    
    print("=" * 70)
    print("MAKING PREDICTIONS ON TEST SET")
    print("=" * 70)
    print()
    
    # ============================================
    # STEP 1: Load Trained Model
    # ============================================
    print("STEP 1: Loading trained model...")
    print("-" * 70)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please train a model first using: python src/train.py")
        return
    
    # Load the model
    # keras.models.load_model() loads a saved model
    model = keras.models.load_model(model_path)
    
    print(f"✓ Model loaded from: {model_path}")
    print(f"  Input shape: {model.input_shape}")
    print()
    
    # ============================================
    # STEP 2: Get List of Test Images
    # ============================================
    print("STEP 2: Scanning test folder...")
    print("-" * 70)
    
    # Get all image files
    image_files = [f for f in os.listdir(test_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sort by filename (assuming they're numbered)
    # This ensures predictions are in order
    image_files.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)
    
    print(f"✓ Found {len(image_files)} test images")
    print()
    
    # ============================================
    # STEP 3: Make Predictions
    # ============================================
    print("STEP 3: Making predictions...")
    print("-" * 70)
    print("This may take a while...")
    print()
    
    # Store predictions
    predictions = []
    image_ids = []
    
    # Process images in batches (faster than one at a time)
    for i in range(0, len(image_files), batch_size):
        # Get batch of filenames
        batch_files = image_files[i:i + batch_size]
        
        # Load and preprocess batch
        batch_images = []
        batch_ids = []
        
        for filename in batch_files:
            image_path = os.path.join(test_folder, filename)
            img_array = preprocess_image(image_path, IMAGE_SIZE)
            batch_images.append(img_array[0])  # Remove batch dimension
            batch_ids.append(int(filename.split('.')[0]))  # Extract ID from filename
        
        # Convert to NumPy array
        batch_images = np.array(batch_images)
        
        # Make predictions
        # model.predict() returns probabilities for each class
        # Shape: (batch_size, num_classes)
        batch_predictions = model.predict(batch_images, verbose=0)
        
        # For binary classification, we get probability of dog (class 1)
        # If sigmoid output: [prob_cat, prob_dog] or just [prob_dog]
        if batch_predictions.shape[1] == 2:
            # Two outputs: take probability of dog (index 1)
            dog_probs = batch_predictions[:, 1]
        else:
            # Single output: probability of dog
            dog_probs = batch_predictions[:, 0]
        
        # Store predictions
        predictions.extend(dog_probs)
        image_ids.extend(batch_ids)
        
        # Print progress
        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(image_files):
            print(f"  Processed {min(i + batch_size, len(image_files))}/{len(image_files)} images...")
    
    print()
    print("✓ Predictions complete")
    print()
    
    # ============================================
    # STEP 4: Save Predictions to CSV
    # ============================================
    print("STEP 4: Saving predictions to CSV...")
    print("-" * 70)
    
    # Create DataFrame (table) with predictions
    # Pandas DataFrame is like an Excel spreadsheet
    df = pd.DataFrame({
        'id': image_ids,           # Image ID
        'label': predictions       # Probability of dog (1 = dog, 0 = cat)
    })
    
    # Sort by ID to ensure correct order
    df = df.sort_values('id').reset_index(drop=True)
    
    # Save to CSV file
    df.to_csv(output_csv, index=False)
    
    print(f"✓ Predictions saved to: {output_csv}")
    print(f"  Total predictions: {len(predictions)}")
    print()
    
    # ============================================
    # STEP 5: Show Statistics
    # ============================================
    print("STEP 5: Prediction statistics...")
    print("-" * 70)
    
    # Convert probabilities to binary predictions (0 or 1)
    # Threshold: > 0.5 = dog, <= 0.5 = cat
    binary_predictions = (np.array(predictions) > 0.5).astype(int)
    
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
    print("You can now submit this file to Kaggle or use it for evaluation.")
    print()


# ============================================
# FUNCTION: Predict Single Image (for testing)
# ============================================
def predict_single_image(model_path, image_path):
    """
    Make prediction on a single image (useful for testing).
    
    Parameters:
    ----------
    model_path : str
        Path to trained model
    image_path : str
        Path to image file
    
    Returns:
    -------
    prediction : float
        Probability that image is a dog (0-1)
    """
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Preprocess image
    img_array = preprocess_image(image_path, IMAGE_SIZE)
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)
    
    # Get dog probability
    if prediction.shape[1] == 2:
        dog_prob = prediction[0, 1]
    else:
        dog_prob = prediction[0, 0]
    
    # Print result
    if dog_prob > 0.5:
        print(f"Prediction: DOG (confidence: {dog_prob*100:.1f}%)")
    else:
        print(f"Prediction: CAT (confidence: {(1-dog_prob)*100:.1f}%)")
    
    return dog_prob


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    """
    Run this script to make predictions:
    python src/predict.py
    """
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print("=" * 70)
        print("ERROR: Model file not found!")
        print("=" * 70)
        print(f"Looking for: {MODEL_PATH}")
        print()
        print("Please:")
        print("1. Train a model first: python src/train.py")
        print("2. Update MODEL_PATH in this file to point to your trained model")
        print()
        models_dir = os.path.join(PROJECT_ROOT, 'models')
        print(f"Available model files in '{models_dir}' folder:")
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
            if model_files:
                for f in model_files:
                    print(f"  - {os.path.join(models_dir, f)}")
            else:
                print("  (No model files found)")
        exit(1)
    
    # Make predictions on test set
    predict_test_set(
        model_path=MODEL_PATH,
        test_folder=TEST_FOLDER,
        output_csv=OUTPUT_CSV,
        batch_size=32
    )

