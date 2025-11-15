# ============================================
# DATA_LOADER.PY - Load and Preprocess Images
# ============================================
# This module provides utilities for loading and preprocessing image data
# for machine learning tasks. It handles image loading, resizing, normalization,
# and data splitting operations.
#
# Image preprocessing transforms raw image files into numerical arrays that
# neural networks can process. The main transformations include:
# - Resizing images to a standard size (e.g., 224x224 pixels)
# - Converting pixel values from 0-255 range to 0-1 range (normalization)
# - Converting images to RGB format (3 color channels)
# - Extracting labels from filenames (0 for cat, 1 for dog)
# ============================================

# Standard library imports for file operations and system functions
import os  # Operating system interface for file and directory operations
import sys  # System-specific parameters for output flushing

# Scientific computing library for numerical arrays
import numpy as np  # NumPy provides array operations and mathematical functions

# Image processing library
from PIL import Image  # Pillow library for opening, manipulating, and saving images

# Plotting library for visualization
import matplotlib.pyplot as plt  # Matplotlib for creating plots and displaying images

# Get project root directory
# This calculates the absolute path to the project root by going up one level
# from the src/ directory where this file is located
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================
# FUNCTION: Load Images from Directory
# ============================================
def load_images_from_folder(folder_path, target_size=(224, 224), max_images=None):
    """
    Load images from a folder and preprocess them for machine learning.
    
    This function reads image files from disk, transforms them into numerical
    arrays, and extracts labels. The preprocessing steps ensure all images have
    the same dimensions and pixel value ranges that neural networks expect.
    
    Processing steps for each image:
    1. Open image file from disk using Pillow library
    2. Convert to RGB format (ensures 3 color channels: Red, Green, Blue)
    3. Resize to target_size (e.g., 224x224 pixels)
    4. Convert to NumPy array with pixel values in range 0-255
    5. Normalize pixel values to range 0-1 by dividing by 255.0
    6. Extract label from filename (0 for cat, 1 for dog)
    
    Parameters:
    ----------
    folder_path : str
        Path to folder containing image files. The folder should contain
        image files with names starting with 'cat' or 'dog' followed by
        a number and extension (e.g., 'cat.0.jpg', 'dog.1234.jpg')
    target_size : tuple, optional
        Target dimensions for resizing images as (width, height).
        Default is (224, 224) which is a standard size used by many
        pre-trained neural networks. Changing this value affects:
        - Memory usage: larger images require more memory
        - Training speed: larger images take longer to process
        - Model accuracy: some architectures work better with specific sizes
    max_images : int or None, optional
        Maximum number of images to load. If None, all images in the folder
        are loaded. If an integer is provided, only the first N images are
        loaded. Useful for:
        - Quick testing with smaller datasets
        - Reducing memory usage during development
        - Debugging data loading issues
    
    Returns:
    -------
    images : numpy.ndarray
        Array containing all preprocessed images. Shape is (num_images, height,
        width, channels). For example, if loading 1000 images at 224x224:
        shape = (1000, 224, 224, 3)
        - First dimension: number of images
        - Second dimension: image height in pixels
        - Third dimension: image width in pixels
        - Fourth dimension: color channels (3 for RGB: Red, Green, Blue)
        Pixel values are normalized to range [0.0, 1.0]
    labels : numpy.ndarray
        Array containing labels for each image. Shape is (num_images,).
        Values are integers: 0 for cat, 1 for dog, -1 for unknown.
        Example: [0, 1, 0, 1, 1, 0, ...] for a dataset with cats and dogs
    filenames : list
        List of strings containing the original filenames for each image.
        Order matches the order in images and labels arrays.
        Example: ['cat.0.jpg', 'dog.1.jpg', 'cat.2.jpg', ...]
    """
    
    # Initialize empty lists to store processed data
    # Lists are used initially because the number of images is unknown
    # They will be converted to NumPy arrays at the end for efficiency
    images = []  # Will store image arrays after preprocessing
    labels = []  # Will store labels (0=cat, 1=dog)
    filenames = []  # Will store original filenames for reference
    
    # Get list of all files in the specified folder
    # os.listdir() returns all files and subdirectories in the folder
    file_list = os.listdir(folder_path)
    
    # Filter file list to include only image files
    # List comprehension creates a new list with files ending in image extensions
    # .lower() converts filename to lowercase for case-insensitive matching
    # .endswith() checks if filename ends with any of the specified extensions
    image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Limit the number of images if max_images parameter is specified
    # This is useful for testing with smaller datasets or limited memory
    # Slicing [:max_images] takes only the first N files from the list
    if max_images:
        image_files = image_files[:max_images]
    
    # Display progress information
    print(f"Loading {len(image_files)} images from {folder_path}...")
    
    # Process each image file one by one
    # enumerate() provides both the index and filename for each iteration
    for idx, filename in enumerate(image_files):
        # Construct full path to image file by joining folder path and filename
        # os.path.join() handles path separators correctly across operating systems
        image_path = os.path.join(folder_path, filename)
        
        # Use try-except block to handle errors gracefully
        # If one image fails to load, processing continues with the next image
        try:
            # Step 1: Open image file using Pillow library
            # Image.open() reads the image file from disk into memory
            # The image is stored as a PIL Image object at this point
            img = Image.open(image_path)
            
            # Step 2: Convert image to RGB format if needed
            # Some images might be in grayscale (1 channel) or other formats
            # Neural networks typically expect RGB images (3 channels)
            # img.mode contains the current color mode (e.g., 'RGB', 'L' for grayscale)
            if img.mode != 'RGB':
                # .convert('RGB') transforms the image to RGB format
                # Grayscale images are duplicated across all 3 channels
                img = img.convert('RGB')
            
            # Step 3: Resize image to target dimensions
            # Neural networks require all input images to have the same size
            # Resizing is necessary because original images may have different dimensions
            # Image.Resampling.LANCZOS is a high-quality resampling algorithm
            # that produces smooth results when scaling images
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Step 4: Convert PIL Image object to NumPy array
            # np.array() converts the image to a NumPy array
            # The array has shape (height, width, 3) with pixel values 0-255
            # .astype('float32') converts integers to 32-bit floating point numbers
            # Division by 255.0 normalizes pixel values from [0, 255] to [0.0, 1.0]
            # Normalization helps neural networks train faster and more stably
            # Example: pixel value 128 becomes 128/255.0 = 0.502
            img_array = np.array(img).astype('float32') / 255.0
            
            # Step 5: Store processed image in the list
            # The image array is appended to the images list
            images.append(img_array)
            # Store the original filename for reference
            filenames.append(filename)
            
            # Step 6: Extract label from filename
            # Labels are determined by checking if filename starts with 'cat' or 'dog'
            # This assumes filenames follow the pattern: "cat.*.jpg" or "dog.*.jpg"
            # .lower() makes the comparison case-insensitive
            if filename.lower().startswith('cat'):
                # Label 0 represents cat images
                labels.append(0)
            elif filename.lower().startswith('dog'):
                # Label 1 represents dog images
                labels.append(1)
            else:
                # Unknown label for files that don't match expected pattern
                labels.append(-1)
            
            # Display progress every 500 images
            # This provides feedback during long loading operations
            # flush=True ensures output appears immediately in the console
            if (idx + 1) % 500 == 0:
                print(f"  Loaded {idx + 1}/{len(image_files)} images...", flush=True)
                sys.stdout.flush()
        
        except Exception as e:
            # If an error occurs while loading an image, print error message
            # and continue processing the next image
            # This prevents one corrupted image from stopping the entire process
            print(f"Error loading {filename}: {e}")
            continue
    
    # Convert Python lists to NumPy arrays for efficient numerical operations
    # NumPy arrays are faster and use less memory than Python lists for numerical data
    # np.array() creates a multi-dimensional array from the list of image arrays
    images = np.array(images)
    # Convert labels list to NumPy array
    labels = np.array(labels)
    
    # Display summary information about loaded data
    print(f"Successfully loaded {len(images)} images", flush=True)
    # Print array shapes to verify data structure
    # images.shape shows (num_images, height, width, channels)
    print(f"Image shape: {images.shape}")
    # labels.shape shows (num_images,)
    print(f"Labels shape: {labels.shape}")
    sys.stdout.flush()
    
    # Return the processed data as a tuple
    return images, labels, filenames


# ============================================
# FUNCTION: Split Data into Train/Validation
# ============================================
def split_data(images, labels, validation_split=0.2, random_seed=42):
    """
    Split data into training and validation sets for machine learning.
    
    This function randomly divides the dataset into two parts:
    - Training set: used to teach the model (typically 80% of data)
    - Validation set: used to evaluate model performance during training (typically 20%)
    
    The split is performed randomly to ensure both sets contain a representative
    mix of cats and dogs. The random seed ensures the same split is produced
    every time the function is called with the same seed value, which is important
    for reproducible experiments.
    
    Process:
    1. Create array of indices [0, 1, 2, ..., num_samples-1]
    2. Shuffle indices randomly to randomize the order
    3. Calculate split point based on validation_split ratio
    4. Use first portion for training, remaining portion for validation
    5. Extract images and labels for each set using the indices
    
    Parameters:
    ----------
    images : numpy.ndarray
        Array containing all image data. Shape is (num_images, height, width, channels).
        This is the complete dataset before splitting.
    labels : numpy.ndarray
        Array containing all labels. Shape is (num_images,).
        Values are 0 for cat, 1 for dog. Must have same length as images.
    validation_split : float, optional
        Fraction of data to reserve for validation set. Must be between 0.0 and 1.0.
        Default is 0.2, meaning 20% of data goes to validation, 80% to training.
        Changing this value affects:
        - More validation data (e.g., 0.3): better evaluation but less training data
        - Less validation data (e.g., 0.1): more training data but less reliable evaluation
        Typical values range from 0.1 to 0.3
    random_seed : int, optional
        Seed value for random number generator. Default is 42.
        Setting the same seed produces the same random split every time.
        This is important for:
        - Reproducible experiments (same split across different runs)
        - Comparing different models on the same data split
        - Debugging and testing (predictable behavior)
        Different seed values produce different random splits
    
    Returns:
    -------
    X_train : numpy.ndarray
        Training set images. Shape is (num_train, height, width, channels).
        Contains approximately (1 - validation_split) * num_images images.
        Example: if 1000 images with validation_split=0.2, X_train has ~800 images
    X_val : numpy.ndarray
        Validation set images. Shape is (num_val, height, width, channels).
        Contains approximately validation_split * num_images images.
        Example: if 1000 images with validation_split=0.2, X_val has ~200 images
    y_train : numpy.ndarray
        Training set labels. Shape is (num_train,).
        Contains labels corresponding to X_train images.
        Values are 0 for cat, 1 for dog.
    y_val : numpy.ndarray
        Validation set labels. Shape is (num_val,).
        Contains labels corresponding to X_val images.
        Values are 0 for cat, 1 for dog.
    """
    
    # Set random seed for NumPy's random number generator
    # This ensures that the random shuffle produces the same result every time
    # when the same seed value is used
    np.random.seed(random_seed)
    
    # Get total number of samples in the dataset
    # len() returns the length of the first dimension of the array
    num_samples = len(images)
    
    # Create array of indices from 0 to num_samples-1
    # np.arange(num_samples) creates [0, 1, 2, ..., num_samples-1]
    # These indices will be used to reference specific images in the dataset
    indices = np.arange(num_samples)
    
    # Shuffle the indices randomly
    # np.random.shuffle() randomly reorders the array in-place
    # After shuffling, indices might be [45, 12, 789, 3, 234, ...] instead of [0, 1, 2, ...]
    # This randomizes which images go to training vs validation
    np.random.shuffle(indices)
    
    # Calculate the index where the split should occur
    # If validation_split = 0.2, then (1 - 0.2) = 0.8 of data goes to training
    # Example: 1000 images * 0.8 = 800, so first 800 indices are for training
    # int() converts to integer to use as array index
    split_idx = int(num_samples * (1 - validation_split))
    
    # Split the shuffled indices into two groups
    # train_indices contains indices for training set (first portion)
    # val_indices contains indices for validation set (remaining portion)
    # Slicing [:] creates views of the array without copying data
    train_indices = indices[:split_idx]  # First portion: [0, 1, ..., split_idx-1]
    val_indices = indices[split_idx:]   # Remaining portion: [split_idx, split_idx+1, ..., end]
    
    # Use the indices to extract corresponding images and labels
    # NumPy array indexing allows selecting specific rows using an array of indices
    # X_train contains images at positions specified by train_indices
    X_train = images[train_indices]
    # X_val contains images at positions specified by val_indices
    X_val = images[val_indices]
    # y_train contains labels at positions specified by train_indices
    y_train = labels[train_indices]
    # y_val contains labels at positions specified by val_indices
    y_val = labels[val_indices]
    
    # Display split information
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} images")
    print(f"  Validation: {len(X_val)} images")
    
    # Return the four arrays as a tuple
    return X_train, X_val, y_train, y_val


# ============================================
# FUNCTION: Visualize Sample Images
# ============================================
def visualize_samples(images, labels, num_samples=8):
    """
    Display sample images with their labels in a grid layout.
    
    This function creates a visualization showing randomly selected images
    from the dataset along with their corresponding labels. Useful for
    verifying that images loaded correctly and understanding the dataset.
    
    The visualization creates a 2x4 grid (8 images total by default) with
    each image labeled as either 'Cat' or 'Dog' based on its label value.
    
    Parameters:
    ----------
    images : numpy.ndarray
        Array containing image data. Shape is (num_images, height, width, channels).
        Images should have pixel values in range [0.0, 1.0] (normalized).
        If images are in range [0, 255], they will appear very bright in the plot.
    labels : numpy.ndarray
        Array containing labels for each image. Shape is (num_images,).
        Values should be 0 for cat, 1 for dog. Must have same length as images.
    num_samples : int, optional
        Number of sample images to display. Default is 8.
        The function creates a grid layout automatically based on this number.
        If num_samples is 8, creates a 2x4 grid (2 rows, 4 columns).
        Changing this value affects the grid layout:
        - 4 samples: 2x2 grid
        - 9 samples: 3x3 grid
        - 16 samples: 4x4 grid
        Maximum recommended is 16 for readability
    
    Returns:
    -------
    None
        This function displays the plot but does not return any value.
        The plot window appears on screen when plt.show() is called.
    """
    
    # Create a figure with subplot grid for displaying multiple images
    # plt.subplots(2, 4) creates a figure with 2 rows and 4 columns of subplots
    # figsize=(12, 6) sets the figure size in inches (width=12, height=6)
    # fig is the entire figure object, axes is a 2D array of subplot axes
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    # Flatten the 2D axes array into a 1D array for easier iteration
    # axes.ravel() converts [[ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]]
    # into [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    axes = axes.ravel()
    
    # Randomly select indices of images to display
    # np.random.choice() selects num_samples random indices from range [0, len(images))
    # replace=False ensures each image is selected at most once (no duplicates)
    # This creates a random sample of images from the dataset
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # Iterate through selected indices and corresponding subplot axes
    # zip() pairs each index with its corresponding subplot axis
    for idx, ax in zip(indices, axes):
        # Display the image in the subplot
        # images[idx] selects the image at position idx from the array
        # ax.imshow() displays the image in the specified subplot
        # imshow() automatically handles the color mapping for RGB images
        ax.imshow(images[idx])
        
        # Hide axis ticks and labels for cleaner appearance
        # ax.axis('off') removes the x and y axis lines and labels
        ax.axis('off')
        
        # Set subplot title based on the label value
        # Ternary operator: 'Cat' if label is 0, 'Dog' if label is 1
        label_text = 'Cat' if labels[idx] == 0 else 'Dog'
        # ax.set_title() adds text above the image
        ax.set_title(label_text, fontsize=10)
    
    # Adjust spacing between subplots for better layout
    # plt.tight_layout() automatically adjusts spacing to prevent overlap
    plt.tight_layout()
    
    # Display the figure in a window
    # plt.show() opens a window displaying the plot
    # The window remains open until closed manually
    plt.show()


# ============================================
# MAIN EXECUTION (runs when script is executed directly)
# ============================================
if __name__ == "__main__":
    """
    Main execution block that runs when this script is executed directly.
    
    This block is useful for testing the data loading functions independently.
    It loads a small sample of images, displays statistics, and visualizes
    sample images to verify that the data loading process works correctly.
    
    To run this script directly:
        python src/data_loader.py
    
    This is helpful for:
    - Testing data loading functionality
    - Verifying image preprocessing works correctly
    - Checking that labels are extracted properly
    - Debugging data loading issues
    """
    
    print("=" * 50)
    print("Testing Data Loader")
    print("=" * 50)
    
    # Construct path to training data folder
    # os.path.join() safely combines path components
    # This creates a path like: /project_root/data/train
    data_train_path = os.path.join(PROJECT_ROOT, 'data', 'train')
    
    # Load a small sample of images for testing
    # max_images=100 limits loading to 100 images for faster testing
    # This is much faster than loading the full dataset (25,000 images)
    # The function returns three values: images array, labels array, and filenames list
    images, labels, filenames = load_images_from_folder(
        folder_path=data_train_path,
        target_size=(224, 224),  # Resize all images to 224x224 pixels
        max_images=100  # Load only 100 images for quick testing
    )
    
    # Calculate and display dataset statistics
    # These statistics help verify that data loaded correctly
    print(f"\nDataset Statistics:")
    # Total number of images loaded
    print(f"  Total images: {len(images)}")
    # Count images labeled as cat (label = 0)
    # np.sum() counts True values in boolean array (labels == 0)
    print(f"  Cats (label 0): {np.sum(labels == 0)}")
    # Count images labeled as dog (label = 1)
    print(f"  Dogs (label 1): {np.sum(labels == 1)}")
    # Display shape of images array
    # Shape shows (num_images, height, width, channels)
    print(f"  Image shape: {images.shape}")
    
    # Visualize randomly selected sample images
    # This displays 8 images in a 2x4 grid with their labels
    print("\nDisplaying sample images...")
    visualize_samples(images, labels, num_samples=8)

