# ============================================
# MODEL.PY - Define Neural Network Architecture
# ============================================
# This module defines neural network model architectures for image classification.
# Four model types are available: simple CNN, advanced CNN, deep custom CNN,
# and transfer learning. Each model type offers different complexity and accuracy.
#
# Neural Network Architecture Overview:
# - Input Layer: Receives preprocessed image data (e.g., 224x224x3 array)
# - Convolutional Layers: Detect spatial patterns like edges, shapes, textures
# - Pooling Layers: Reduce spatial dimensions while preserving important features
# - Dense Layers: Perform classification based on extracted features
# - Output Layer: Produces final prediction (probability of dog vs cat)
#
# Model Type Comparison:
# - Simple CNN: 3 convolutional blocks, trains from scratch, moderate accuracy
# - Advanced CNN: 4 convolutional blocks, trains from scratch, better accuracy
# - Deep Custom CNN: 5+ convolutional blocks, trains from scratch, highest accuracy potential
# - Transfer Learning: Pre-trained base model, fine-tuned, best accuracy and speed
# ============================================

# TensorFlow and Keras imports for deep learning functionality
# TensorFlow is the underlying deep learning framework
# Keras provides high-level API for building and training neural networks
import tensorflow as tf  # TensorFlow library for numerical computation and machine learning
from tensorflow import keras  # Keras API for building neural network models
from tensorflow.keras import layers, models  # Layer types and model architectures

# ============================================
# FUNCTION: Create Simple CNN Model
# ============================================
def create_simple_cnn_model(input_shape=(224, 224, 3), num_classes=2, config=None):
    """
    Creates a simple Convolutional Neural Network (CNN) for image classification.
    
    This model consists of three convolutional blocks followed by fully connected
    layers. Suitable for learning the basics of CNNs and moderate accuracy tasks.
    The architecture trains from scratch without pre-trained weights.
    
    Architecture Flow:
    1. Input: Image array of shape (224, 224, 3) with normalized pixel values [0, 1]
    2. Conv Block 1: 32 filters, detects basic edges and textures
    3. Conv Block 2: 64 filters, detects more complex patterns
    4. Conv Block 3: 128 filters, detects high-level features
    5. Flatten: Converts 2D feature maps to 1D vector
    6. Dense Layer 1: 512 neurons, processes extracted features
    7. Dense Layer 2: 256 neurons, further feature processing
    8. Output: Single neuron with sigmoid activation, outputs probability [0, 1]
    
    Each convolutional block consists of:
    - Conv2D layer: Applies convolution operation to detect patterns
    - MaxPooling2D layer: Reduces spatial dimensions by factor of 2
    
    Parameters:
    ----------
    input_shape : tuple, optional
        Shape of input images as (height, width, channels). Default is (224, 224, 3)
        for 224x224 pixel RGB images. Changing this affects:
        - Memory usage: larger images require more memory
        - Training speed: larger images process slower
        - Model architecture: must match actual image dimensions
    num_classes : int, optional
        Number of output classes. Default is 2 for binary classification (cat/dog).
        For binary classification, output layer has 1 neuron with sigmoid activation.
        For multi-class (3+ classes), would use num_classes neurons with softmax.
    config : dict, optional
        Configuration dictionary containing model hyperparameters:
        - conv_blocks: Number of convolutional blocks (default: 3)
          More blocks = deeper network = more complex features but slower training
        - filters: List of filter counts per block (default: [32, 64, 128])
          Each number represents how many feature detectors in that block
          Increasing filters = more features detected but more parameters
        - dense_units: List of dense layer units (default: [512, 256])
          Number of neurons in each fully connected layer
          More units = more capacity but risk of overfitting
        - dropout_rate: Dropout rate for regularization (default: 0.5)
          Fraction of neurons randomly disabled during training (0.0 to 1.0)
          Higher rate = more regularization = less overfitting but slower learning
        - learning_rate: Learning rate for optimizer (default: 0.001)
          Controls step size during training (typically 0.0001 to 0.01)
          Higher rate = faster learning but may overshoot optimal values
    
    Returns:
    -------
    model : keras.Model
        Compiled neural network model ready for training. The model has:
        - Architecture defined with all layers
        - Optimizer configured (Adam with specified learning rate)
        - Loss function set (binary_crossentropy for binary classification)
        - Metrics tracked (accuracy during training)
    
    Notes:
    -----
    - Convolutional layers detect spatial patterns in images using learned filters
    - Max pooling layers reduce spatial dimensions by taking maximum value in each region
    - Dense layers perform classification based on extracted features
    - Dropout prevents overfitting by randomly setting some weights to zero during training
    - Sigmoid activation in output layer produces probability between 0 and 1
    - Model trains from scratch, requiring more data and time than transfer learning
    """
    if config is None:
        config = {}
    
    # Extract configuration parameters with default values
    # These defaults provide a good starting point for binary image classification
    conv_blocks = config.get('conv_blocks', 3)  # Number of convolutional blocks
    filters = config.get('filters', [32, 64, 128])  # Filters per block (increasing pattern)
    dense_units = config.get('dense_units', [512, 256])  # Neurons in dense layers
    dropout_rate = config.get('dropout_rate', 0.5)  # Dropout regularization rate
    learning_rate = config.get('learning_rate', 0.001)  # Learning rate for optimizer
    
    # Ensure filters list matches the number of convolutional blocks
    # If insufficient filters provided, extend list by doubling last value
    # This ensures each block has an appropriate number of filters
    # Example: if conv_blocks=5 and filters=[32, 64], becomes [32, 64, 128, 256, 512]
    if len(filters) < conv_blocks:
        filters = filters + [filters[-1] * 2] * (conv_blocks - len(filters))
    # Take only the number of filters needed for specified blocks
    filters = filters[:conv_blocks]
    
    # Create Sequential model to stack layers linearly
    # Sequential model is appropriate when layers are added one after another
    # Alternative: Functional API for more complex architectures with branches
    model = models.Sequential()
    
    # Add convolutional blocks
    # Each block consists of Conv2D layer followed by MaxPooling2D layer
    # enumerate() provides both index and filter count for each block
    for i, num_filters in enumerate(filters):
        if i == 0:
            # First block requires input_shape parameter to define input dimensions
            # Conv2D: 2D convolution layer for image processing
            # num_filters: number of convolutional filters (feature detectors)
            # (3, 3): kernel size (3x3 pixels) - standard size for detecting small patterns
            # activation='relu': Rectified Linear Unit activation function
            #   ReLU outputs the input if positive, otherwise outputs 0
            #   Helps introduce non-linearity needed for learning complex patterns
            # input_shape: specifies the shape of input data (height, width, channels)
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu', input_shape=input_shape))
        else:
            # Subsequent blocks do not need input_shape (automatically inferred)
            # Each block typically has more filters than the previous one
            # This allows detecting increasingly complex features at deeper layers
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu'))
        
        # MaxPooling2D layer reduces spatial dimensions by factor of 2
        # (2, 2): pool size - takes maximum value from each 2x2 region
        # Benefits: reduces computation, prevents overfitting, makes features more robust
        # After pooling, image dimensions are halved (e.g., 224x224 -> 112x112)
        model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten layer converts 2D feature maps to 1D vector for dense layers
    # After convolutional blocks, data is still 2D (height x width x channels)
    # Dense layers require 1D input, so flattening is necessary
    # Example: if last conv output is (14, 14, 128), flatten produces (25088,)
    model.add(layers.Flatten())
    
    # Add fully connected dense layers
    # Dense layers perform final classification based on extracted features
    # Each dense layer processes all features from previous layer
    for units in dense_units:
        # Dense layer with ReLU activation for non-linearity
        # units: number of neurons in this layer
        # More neurons = more capacity to learn complex patterns
        # ReLU activation helps with gradient flow during backpropagation
        model.add(layers.Dense(units, activation='relu'))
        
        # Dropout layer for regularization
        # Randomly sets a fraction (dropout_rate) of inputs to zero during training
        # Prevents overfitting by forcing model to not rely on specific neurons
        # During inference (prediction), all neurons are active
        # dropout_rate=0.5 means 50% of neurons are randomly disabled each training step
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer for binary classification
    # Single neuron with sigmoid activation outputs probability between 0 and 1
    # Sigmoid function maps any input to range (0, 1)
    # Output interpretation: close to 0 = cat, close to 1 = dog
    # Threshold of 0.5 typically used to make binary decision
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model with optimizer, loss function, and metrics
    # Compilation prepares the model for training by configuring:
    # - How the model learns (optimizer)
    # - How to measure errors (loss function)
    # - What metrics to track during training
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        # Adam optimizer: adaptive learning rate algorithm
        # Combines benefits of AdaGrad and RMSProp optimizers
        # learning_rate controls step size: higher = faster but less stable
        loss='binary_crossentropy',  # Binary classification loss function
        # Measures difference between predicted and actual probabilities
        # Lower loss = better predictions
        metrics=['accuracy']  # Track accuracy during training
        # Accuracy = fraction of correct predictions (0.0 to 1.0)
    )
    
    # Display model architecture summary
    # Shows layer types, output shapes, and parameter counts
    # Useful for understanding model structure and debugging
    print("=" * 50)
    print("Simple CNN Model Architecture:")
    print("=" * 50)
    model.summary()
    
    return model


# ============================================
# FUNCTION: Create Advanced CNN Model
# ============================================
def create_advanced_cnn_model(input_shape=(224, 224, 3), num_classes=2, config=None):
    """
    Creates an advanced Convolutional Neural Network with deeper architecture.
    
    This model uses four convolutional blocks with more filters and larger dense
    layers compared to the simple CNN. Provides better feature extraction and
    typically achieves higher accuracy, but requires more training time and
    computational resources.
    
    Architecture Flow:
    1. Input: Image array of shape (224, 224, 3) with normalized pixel values [0, 1]
    2. Conv Block 1: 32 filters, detects basic edges and textures
    3. Conv Block 2: 64 filters, detects more complex patterns
    4. Conv Block 3: 128 filters, detects high-level features
    5. Conv Block 4: 256 filters, detects very complex patterns
    6. Flatten: Converts 2D feature maps to 1D vector
    7. Dense Layer 1: 1024 neurons, processes extracted features
    8. Dense Layer 2: 512 neurons, further feature processing
    9. Output: Single neuron with sigmoid activation, outputs probability [0, 1]
    
    Compared to Simple CNN:
    - One additional convolutional block (4 vs 3)
    - More filters in later blocks (up to 256 vs 128)
    - Larger dense layers (1024 vs 512 neurons)
    - Lower learning rate for stability (0.0001 vs 0.001)
    
    Parameters:
    ----------
    input_shape : tuple, optional
        Shape of input images as (height, width, channels). Default is (224, 224, 3)
        for 224x224 pixel RGB images. Must match the dimensions of preprocessed images.
    num_classes : int, optional
        Number of output classes. Default is 2 for binary classification (cat/dog).
    config : dict, optional
        Configuration dictionary containing model hyperparameters:
        - conv_blocks: Number of convolutional blocks (default: 4)
          More blocks allow learning more complex hierarchical features
        - filters: List of filter counts per block (default: [32, 64, 128, 256])
          Increasing pattern allows detecting features at multiple scales
        - dense_units: List of dense layer units (default: [1024, 512])
          Larger than simple CNN to handle more complex feature combinations
        - dropout_rate: Dropout rate for regularization (default: 0.5)
          Important for preventing overfitting with larger model capacity
        - learning_rate: Learning rate for optimizer (default: 0.0001)
          Lower than simple CNN for more stable training of deeper network
    
    Returns:
    -------
    model : keras.Model
        Compiled neural network model ready for training. The model has:
        - Deeper architecture with 4 convolutional blocks
        - Larger dense layers for complex pattern recognition
        - Optimizer configured with lower learning rate
        - Loss function and metrics set for binary classification
    
    Notes:
    -----
    - Deeper architecture allows learning more complex features than simple CNN
    - More parameters require careful regularization to avoid overfitting
    - Lower learning rate helps with training stability in deeper networks
    - Training time is longer than simple CNN due to increased complexity
    - Typically achieves 5-10% better accuracy than simple CNN
    - Requires more memory and computational resources
    """
    if config is None:
        config = {}
    
    # Extract configuration parameters with default values for advanced architecture
    # Defaults are tuned for better performance with deeper network
    conv_blocks = config.get('conv_blocks', 4)  # Four convolutional blocks
    filters = config.get('filters', [32, 64, 128, 256])  # More filters than simple CNN
    dense_units = config.get('dense_units', [1024, 512])  # Larger dense layers
    dropout_rate = config.get('dropout_rate', 0.5)  # Same dropout rate
    learning_rate = config.get('learning_rate', 0.0001)  # Lower learning rate for stability
    
    # Ensure filters list matches the number of convolutional blocks
    # Extends list if needed by doubling the last value
    if len(filters) < conv_blocks:
        filters = filters + [filters[-1] * 2] * (conv_blocks - len(filters))
    filters = filters[:conv_blocks]
    
    # Create Sequential model for stacking layers
    model = models.Sequential()
    
    # Add convolutional blocks with increasing filter counts
    # Each block extracts features at a different level of abstraction
    for i, num_filters in enumerate(filters):
        if i == 0:
            # First block requires input_shape to define model input dimensions
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu', input_shape=input_shape))
        else:
            # Subsequent blocks automatically infer input shape from previous layer
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu'))
        # Max pooling reduces spatial dimensions after each convolutional block
        model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten 2D feature maps to 1D vector for dense layers
    # Converts multi-dimensional feature maps into a single vector
    model.add(layers.Flatten())
    
    # Add larger fully connected layers for advanced feature learning
    # These layers combine features from all spatial locations
    for units in dense_units:
        # Dense layer with ReLU activation processes extracted features
        model.add(layers.Dense(units, activation='relu'))
        # Dropout layer prevents overfitting by randomly disabling neurons
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer for binary classification
    # Single neuron with sigmoid outputs probability of dog (0 to 1)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model with lower learning rate for stability
    # Deeper networks benefit from more conservative learning rates
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',  # Binary classification loss
        metrics=['accuracy']  # Track accuracy during training
    )
    
    # Display model architecture summary
    print("=" * 50)
    print("Advanced CNN Model Architecture:")
    print("=" * 50)
    model.summary()
    
    return model


# ============================================
# FUNCTION: Create Deep Custom CNN Model
# ============================================
def create_deep_custom_cnn_model(input_shape=(224, 224, 3), num_classes=2, config=None):
    """
    Creates a deep custom Convolutional Neural Network with maximum depth.
    
    This model uses five or more convolutional blocks with extensive filters and
    large dense layers. Designed for maximum feature extraction capability and
    highest possible accuracy when training from scratch. Requires significant
    computational resources and training time.
    
    Architecture Flow:
    1. Input: Image array of shape (224, 224, 3) with normalized pixel values [0, 1]
    2. Conv Block 1: 32 filters, detects basic edges and textures
    3. Conv Block 2: 64 filters, detects more complex patterns
    4. Conv Block 3: 128 filters, detects high-level features
    5. Conv Block 4: 256 filters, detects very complex patterns
    6. Conv Block 5: 512 filters, detects extremely complex hierarchical features
    7. Flatten: Converts 2D feature maps to 1D vector
    8. Dense Layer 1: 1024 neurons, processes extracted features
    9. Dense Layer 2: 512 neurons, further feature processing
    10. Dense Layer 3: 256 neurons, final feature refinement
    11. Output: Single neuron with sigmoid activation, outputs probability [0, 1]
    
    Compared to Advanced CNN:
    - One additional convolutional block (5 vs 4)
    - More filters in deepest block (512 vs 256)
    - Additional dense layer (3 vs 2 dense layers)
    - Maximum depth for from-scratch training
    
    Parameters:
    ----------
    input_shape : tuple, optional
        Shape of input images as (height, width, channels). Default is (224, 224, 3)
        for 224x224 pixel RGB images. Must match the dimensions of preprocessed images.
    num_classes : int, optional
        Number of output classes. Default is 2 for binary classification (cat/dog).
    config : dict, optional
        Configuration dictionary containing model hyperparameters:
        - conv_blocks: Number of convolutional blocks (default: 5)
          Maximum depth for from-scratch training without pre-trained weights
        - filters: List of filter counts per block (default: [32, 64, 128, 256, 512])
          Extensive filter progression for maximum feature detection
        - dense_units: List of dense layer units (default: [1024, 512, 256])
          Three dense layers for complex feature combination
        - dropout_rate: Dropout rate for regularization (default: 0.5)
          Critical for preventing overfitting with very large model capacity
        - learning_rate: Learning rate for optimizer (default: 0.0001)
          Conservative learning rate for stable training of deep network
    
    Returns:
    -------
    model : keras.Model
        Compiled neural network model ready for training. The model has:
        - Deepest architecture with 5+ convolutional blocks
        - Extensive filter counts for maximum feature extraction
        - Multiple large dense layers for complex pattern recognition
        - Optimizer configured with conservative learning rate
        - Loss function and metrics set for binary classification
    
    Notes:
    -----
    - Deepest architecture with most parameters among from-scratch models
    - Higher risk of overfitting requires careful regularization
    - May require data augmentation for best results
    - Training time significantly longer than simpler architectures
    - Requires substantial memory and computational resources
    - Typically achieves highest accuracy among from-scratch models
    - Still may not match transfer learning performance
    """
    if config is None:
        config = {}
    
    # Extract configuration parameters with default values for deep architecture
    # Defaults are tuned for maximum depth and feature extraction
    conv_blocks = config.get('conv_blocks', 5)  # Five convolutional blocks
    filters = config.get('filters', [32, 64, 128, 256, 512])  # Extensive filter progression
    dense_units = config.get('dense_units', [1024, 512, 256])  # Three dense layers
    dropout_rate = config.get('dropout_rate', 0.5)  # Dropout for regularization
    learning_rate = config.get('learning_rate', 0.0001)  # Conservative learning rate
    
    # Ensure filters list matches the number of convolutional blocks
    # Extends list if needed by doubling the last value
    if len(filters) < conv_blocks:
        filters = filters + [filters[-1] * 2] * (conv_blocks - len(filters))
    filters = filters[:conv_blocks]
    
    # Create Sequential model for stacking layers
    model = models.Sequential()
    
    # Add multiple convolutional blocks for deep feature extraction
    # Each block extracts increasingly abstract features
    for i, num_filters in enumerate(filters):
        if i == 0:
            # First block requires input_shape to define model input dimensions
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu', input_shape=input_shape))
        else:
            # Subsequent blocks automatically infer input shape from previous layer
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu'))
        # Max pooling reduces spatial dimensions after each convolutional block
        model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten 2D feature maps to 1D vector for dense layers
    # Converts multi-dimensional feature maps into a single vector
    model.add(layers.Flatten())
    
    # Add multiple large dense layers for complex pattern recognition
    # Three dense layers allow for hierarchical feature combination
    for units in dense_units:
        # Dense layer with ReLU activation processes extracted features
        model.add(layers.Dense(units, activation='relu'))
        # Dropout layer prevents overfitting by randomly disabling neurons
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer for binary classification
    # Single neuron with sigmoid outputs probability of dog (0 to 1)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model with conservative learning rate
    # Very deep networks require careful learning rate tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',  # Binary classification loss
        metrics=['accuracy']  # Track accuracy during training
    )
    
    # Display model architecture summary
    print("=" * 50)
    print("Deep Custom CNN Model Architecture:")
    print("=" * 50)
    model.summary()
    
    return model


# ============================================
# FUNCTION: Create Transfer Learning Model
# ============================================
def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=2, config=None):
    """
    Creates a model using transfer learning with a pre-trained base model.
    
    Transfer learning leverages neural networks pre-trained on large datasets
    (typically ImageNet with 1.4 million images and 1000 classes) and adapts
    them for the specific classification task. This approach typically achieves
    better accuracy with less training time compared to training from scratch.
    The base model provides learned features that are fine-tuned for the target task.
    
    Architecture Flow:
    1. Input: Image array of shape (224, 224, 3) with normalized pixel values [0, 1]
    2. Pre-trained Base Model: Extracts features using learned weights from ImageNet
       - MobileNetV2: Efficient architecture with depthwise separable convolutions
       - Alternative: ResNet50, VGG16, InceptionV3, etc.
    3. Global Average Pooling: Reduces spatial dimensions to single vector per channel
    4. Dense Layer: Processes extracted features (typically 128 neurons)
    5. Output: Single neuron with sigmoid activation, outputs probability [0, 1]
    
    Transfer Learning Benefits:
    - Pre-trained weights provide excellent feature extraction from day one
    - Requires less training data than from-scratch models
    - Faster training (fewer epochs needed)
    - Typically achieves best accuracy among all model types
    - Industry standard approach for image classification
    
    Parameters:
    ----------
    input_shape : tuple, optional
        Shape of input images as (height, width, channels). Default is (224, 224, 3)
        for 224x224 pixel RGB images. Must match the input requirements of the
        base model (some models require specific sizes like 224x224 or 299x299).
    num_classes : int, optional
        Number of output classes. Default is 2 for binary classification (cat/dog).
    config : dict, optional
        Configuration dictionary containing model hyperparameters:
        - base_model: Name of pre-trained base model (default: 'MobileNetV2')
          Options: 'MobileNetV2', 'ResNet50', 'VGG16', 'InceptionV3', 'EfficientNetB0', etc.
          MobileNetV2: Fast and efficient, good for mobile devices
          ResNet50: Deeper architecture, higher accuracy potential
          VGG16: Classic architecture, well-understood
        - weights: Pre-trained weights to use (default: 'imagenet')
          'imagenet': Loads weights pre-trained on ImageNet dataset
          None: Random initialization (defeats purpose of transfer learning)
        - trainable_base: Whether to train base model layers (default: False)
          False: Freezes base layers (faster training, less memory, standard approach)
          True: Allows fine-tuning of base layers (slower, more memory, potentially better)
        - dense_units: List of dense layer units (default: [128])
          Smaller than from-scratch models because base model already extracts features
        - dropout_rate: Dropout rate for regularization (default: 0.5)
          Prevents overfitting on custom classification layers
        - learning_rate: Learning rate for optimizer (default: 0.001)
          Can use higher learning rate than from-scratch models
    
    Returns:
    -------
    model : keras.Model
        Compiled neural network model ready for training. The model has:
        - Pre-trained base model for feature extraction
        - Custom classification layers for binary classification
        - Optimizer configured with specified learning rate
        - Loss function and metrics set for binary classification
    
    Notes:
    -----
    - Pre-trained models provide excellent feature extraction from the start
    - Freezing base layers (trainable_base=False) speeds up training significantly
    - Fine-tuning base layers may improve accuracy but increases training time
    - Typically achieves best accuracy with moderate computational cost
    - Requires less training data than from-scratch models
    - Industry standard approach for most image classification tasks
    """
    if config is None:
        config = {}
    
    # Extract configuration parameters with default values
    # Defaults are chosen for good balance of speed and accuracy
    base_model_name = config.get('base_model', 'MobileNetV2')  # Efficient base model
    weights = config.get('weights', 'imagenet')  # Pre-trained ImageNet weights
    trainable_base = config.get('trainable_base', False)  # Freeze base by default
    dense_units = config.get('dense_units', [128])  # Smaller dense layer than from-scratch
    dropout_rate = config.get('dropout_rate', 0.5)  # Standard dropout rate
    learning_rate = config.get('learning_rate', 0.001)  # Can use higher learning rate
    
    # Load pre-trained base model from Keras applications
    # getattr dynamically retrieves the model class by name from keras.applications
    # This allows flexible selection of different base models via configuration
    base_model_class = getattr(keras.applications, base_model_name, None)
    if base_model_class is None:
        # Raise error if specified base model is not available in Keras
        raise ValueError(f"Unknown base model: {base_model_name}. Available models: "
                        f"MobileNetV2, ResNet50, VGG16, InceptionV3, etc.")
    
    # Create base model without top classification layers
    # include_top=False allows adding custom classification layers for our task
    # The base model outputs feature maps, not final predictions
    # weights='imagenet' loads pre-trained weights from ImageNet dataset
    base_model = base_model_class(
        input_shape=input_shape,  # Input image dimensions
        include_top=False,  # Exclude original classification layers
        weights=weights  # Load pre-trained weights
    )
    
    # Control whether base model layers are trainable
    # False: freeze weights (faster training, less memory, standard approach)
    #   Base model weights remain fixed, only custom layers are trained
    #   This is the typical approach for transfer learning
    # True: allow fine-tuning (slower training, potentially better accuracy)
    #   Base model weights can be updated during training
    #   Requires more memory and computation, but may improve results
    base_model.trainable = trainable_base
    
    # Create Sequential model with base model and pooling layer
    # Sequential model stacks layers one after another
    model = models.Sequential([
        base_model,  # Pre-trained feature extractor
        # GlobalAveragePooling2D reduces spatial dimensions to single vector
        # Takes average of each feature map, converting (H, W, C) to (C,)
        # This is more efficient than Flatten and often works better
        layers.GlobalAveragePooling2D(),  # Reduces spatial dimensions to single vector
    ])
    
    # Add custom classification layers
    # These layers learn to classify based on features extracted by base model
    # Typically smaller than from-scratch models because base model already does feature extraction
    for units in dense_units:
        # Dense layer with ReLU activation processes extracted features
        model.add(layers.Dense(units, activation='relu'))
        # Dropout layer prevents overfitting on custom layers
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer for binary classification
    # Single neuron with sigmoid activation outputs probability of dog (0 to 1)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model with optimizer, loss function, and metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        # Adam optimizer with specified learning rate
        # Can use higher learning rate than from-scratch models
        loss='binary_crossentropy',  # Binary classification loss
        metrics=['accuracy']  # Track accuracy during training
    )
    
    # Display model architecture summary
    # Shows base model layers (frozen or trainable) and custom layers
    print("=" * 50)
    print("Transfer Learning Model Architecture:")
    print("=" * 50)
    model.summary()
    
    return model


# ============================================
# MAIN EXECUTION (for testing)
# ============================================
if __name__ == "__main__":
    """
    Main execution block for testing model creation.
    
    This block runs when the script is executed directly (not imported).
    It creates a simple CNN model and displays information about the model
    architecture, including total parameters and input/output shapes.
    
    To run this script directly:
        python src/model.py
    
    This is useful for:
    - Testing model creation functions
    - Verifying model architecture
    - Understanding model complexity (parameter count)
    - Debugging model definition issues
    """
    
    # Create a simple CNN model for testing
    # This demonstrates the basic model creation process
    print("Creating simple CNN model...")
    model = create_simple_cnn_model()
    
    # Display model information
    # This helps understand the model structure and complexity
    print("\n" + "=" * 50)
    print("Model created successfully!")
    print("=" * 50)
    # Total parameters: total number of trainable weights in the model
    # More parameters = more capacity but also more risk of overfitting
    print(f"Total parameters: {model.count_params():,}")
    # Input shape: dimensions of input data the model expects
    print(f"Input shape: {model.input_shape}")
    # Output shape: dimensions of output predictions
    print(f"Output shape: {model.output_shape}")

