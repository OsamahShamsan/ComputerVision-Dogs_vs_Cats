# ============================================
# DATA_LOADER.PY - Load and Preprocess Images
# ============================================
# This script handles:
#   1. Loading images from folders
#   2. Resizing images to a standard size
#   3. Converting images to NumPy arrays
#   4. Creating labels (0 = cat, 1 = dog)
#   5. Splitting data into training and validation sets
# ============================================

# Import necessary libraries
import os                    # Operating system interface (file/folder operations)
import sys                   # System-specific parameters and functions (for stdout flushing)
import numpy as np           # NumPy: numerical computing (arrays, math)
from PIL import Image        # Pillow: image processing (open, resize images)
import matplotlib.pyplot as plt  # Matplotlib: plotting/visualization

# Get project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================
# FUNCTION: Load Images from Directory
# ============================================
def load_images_from_folder(folder_path, target_size=(224, 224), max_images=None):
    """
    Load images from a folder and preprocess them.
    
    Parameters (inputs):
    ----------
    folder_path : str
        Path to folder containing images (e.g., 'train/')
    target_size : tuple
        Size to resize images to (width, height)
        Default: (224, 224) - common size for neural networks
    max_images : int or None
        Maximum number of images to load (None = load all)
        Useful for testing with smaller datasets
    
    Returns (outputs):
    -------
    images : numpy array
        Array of image data (shape: [num_images, height, width, channels])
    labels : numpy array
        Array of labels (0 for cat, 1 for dog)
    filenames : list
        List of image filenames
    """
    
    # Initialize empty lists to store data
    images = []          # Will store image arrays
    labels = []          # Will store labels (0=cat, 1=dog)
    filenames = []       # Will store filenames
    
    # Get list of all files in the folder
    # os.listdir() returns all files/folders in the directory
    file_list = os.listdir(folder_path)
    
    # Filter to only image files (JPG, JPEG, PNG)
    # List comprehension: creates a new list with only image files
    image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Limit number of images if max_images is specified
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Loading {len(image_files)} images from {folder_path}...")
    
    # Loop through each image file
    for idx, filename in enumerate(image_files):
        # Create full path to image file
        # os.path.join() safely combines folder path + filename
        image_path = os.path.join(folder_path, filename)
        
        try:
            # Open image using Pillow
            # Image.open() reads the image file
            img = Image.open(image_path)
            
            # Convert to RGB if needed (some images might be grayscale)
            # Neural networks expect RGB (3 channels: Red, Green, Blue)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image to target size
            # Neural networks need all images to be the same size
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert PIL Image to NumPy array
            # Images are stored as arrays of pixel values (0-255)
            img_array = np.array(img)
            
            # Normalize pixel values to range [0, 1]
            # Neural networks work better with values between 0 and 1
            # Original values are 0-255, so divide by 255.0
            img_array = img_array.astype('float32') / 255.0
            
            # Add image to list
            images.append(img_array)
            filenames.append(filename)
            
            # Determine label from filename
            # In this dataset, filenames are like: "cat.0.jpg" or "dog.1234.jpg"
            if filename.startswith('cat'):
                labels.append(0)  # 0 = cat
            elif filename.startswith('dog'):
                labels.append(1)  # 1 = dog
            else:
                # If filename doesn't start with cat/dog, skip or set default
                labels.append(-1)  # Unknown label
            
            # Print progress every 500 images (more frequent updates)
            if (idx + 1) % 500 == 0:
                print(f"  Loaded {idx + 1}/{len(image_files)} images...", flush=True)
                sys.stdout.flush()  # Force output to appear immediately
        
        except Exception as e:
            # If there's an error loading an image, skip it and continue
            print(f"Error loading {filename}: {e}")
            continue
    
    # Convert lists to NumPy arrays
    # NumPy arrays are more efficient for numerical operations
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Successfully loaded {len(images)} images", flush=True)
    print(f"Image shape: {images.shape}")  # Should be (num_images, height, width, 3)
    print(f"Labels shape: {labels.shape}")  # Should be (num_images,)
    sys.stdout.flush()
    
    return images, labels, filenames


# ============================================
# FUNCTION: Split Data into Train/Validation
# ============================================
def split_data(images, labels, validation_split=0.2, random_seed=42):
    """
    Split data into training and validation sets.
    
    Parameters:
    ----------
    images : numpy array
        All image data
    labels : numpy array
        All labels
    validation_split : float
        Fraction of data to use for validation (0.2 = 20%)
    random_seed : int
        Random seed for reproducibility (same seed = same split)
    
    Returns:
    -------
    X_train, X_val, y_train, y_val : numpy arrays
        Training and validation sets
    """
    
    # Set random seed for reproducibility
    # This ensures we get the same split every time
    np.random.seed(random_seed)
    
    # Get number of samples
    num_samples = len(images)
    
    # Create array of indices [0, 1, 2, ..., num_samples-1]
    indices = np.arange(num_samples)
    
    # Shuffle indices randomly
    # This randomizes the order of images
    np.random.shuffle(indices)
    
    # Calculate split point
    # If validation_split = 0.2, use 20% for validation
    split_idx = int(num_samples * (1 - validation_split))
    
    # Split indices into train and validation
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Use indices to split images and labels
    X_train = images[train_indices]
    X_val = images[val_indices]
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} images")
    print(f"  Validation: {len(X_val)} images")
    
    return X_train, X_val, y_train, y_val


# ============================================
# FUNCTION: Visualize Sample Images
# ============================================
def visualize_samples(images, labels, num_samples=8):
    """
    Display sample images with their labels.
    
    Parameters:
    ----------
    images : numpy array
        Image data
    labels : numpy array
        Labels (0=cat, 1=dog)
    num_samples : int
        Number of samples to display
    """
    
    # Create a figure with subplots
    # plt.subplots() creates a grid of plots
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()  # Flatten 2D array to 1D for easier iteration
    
    # Display random samples
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    for idx, ax in zip(indices, axes):
        # Display image
        # images[idx] is one image array
        # imshow() displays the image
        ax.imshow(images[idx])
        ax.axis('off')  # Hide axes
        
        # Set title based on label
        label_text = 'Cat' if labels[idx] == 0 else 'Dog'
        ax.set_title(label_text, fontsize=10)
    
    plt.tight_layout()  # Adjust spacing
    plt.show()  # Display the plot


# ============================================
# MAIN EXECUTION (runs when script is executed directly)
# ============================================
if __name__ == "__main__":
    """
    This block runs only when you execute this file directly:
    python src/data_loader.py
    
    It's useful for testing the functions.
    """
    
    print("=" * 50)
    print("Testing Data Loader")
    print("=" * 50)
    
    # Load a small sample of images for testing
    # max_images=100 means we'll only load 100 images (faster for testing)
    data_train_path = os.path.join(PROJECT_ROOT, 'data', 'train')
    images, labels, filenames = load_images_from_folder(
        folder_path=data_train_path,
        target_size=(224, 224),
        max_images=100  # Small number for quick testing
    )
    
    # Print some statistics
    print(f"\nDataset Statistics:")
    print(f"  Total images: {len(images)}")
    print(f"  Cats (label 0): {np.sum(labels == 0)}")
    print(f"  Dogs (label 1): {np.sum(labels == 1)}")
    print(f"  Image shape: {images.shape}")
    
    # Visualize some samples
    print("\nDisplaying sample images...")
    visualize_samples(images, labels, num_samples=8)

