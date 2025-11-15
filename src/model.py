# ============================================
# MODEL.PY - Define Neural Network Architecture
# ============================================
# This script creates the neural network model structure.
# 
# A neural network is like a brain with layers:
#   - Input Layer: Receives the image data
#   - Hidden Layers: Process and learn patterns
#   - Output Layer: Makes the prediction (dog or cat)
# ============================================

# Import TensorFlow and Keras
# TensorFlow: Deep learning framework
# Keras: High-level API that makes building models easier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# ============================================
# FUNCTION: Create a Simple CNN Model
# ============================================
def create_simple_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a Convolutional Neural Network (CNN) for image classification.
    
    CNN Explanation:
    - CNNs are designed for images
    - They use "convolution" to detect patterns (edges, shapes, features)
    - Multiple layers learn increasingly complex patterns
    - Final layers make the classification decision
    
    Parameters:
    ----------
    input_shape : tuple
        Shape of input images (height, width, channels)
        Default: (224, 224, 3) = 224x224 pixels, 3 color channels (RGB)
    num_classes : int
        Number of output classes (2 = dog or cat)
    
    Returns:
    -------
    model : Keras Model
        Compiled neural network model ready for training
    """
    
    # Create a Sequential model
    # Sequential = layers stacked one after another (like building blocks)
    model = models.Sequential([
        
        # ============================================
        # CONVOLUTIONAL LAYERS (Feature Detection)
        # ============================================
        # These layers detect patterns in images
        
        # First Convolutional Block
        # Conv2D: Applies filters to detect features (edges, textures)
        # 32 filters: Creates 32 different feature maps
        # (3, 3): Filter size (3x3 pixels)
        # activation='relu': ReLU activation function (makes negative values 0)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        
        # MaxPooling2D: Reduces image size, keeps important features
        # (2, 2): Takes maximum value from 2x2 regions
        # This makes the model faster and reduces overfitting
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        # 64 filters: More filters = can detect more complex patterns
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        # 128 filters: Even more complex pattern detection
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # ============================================
        # FLATTEN LAYER
        # ============================================
        # Converts 2D feature maps to 1D array
        # Example: (7, 7, 128) becomes (6272,)
        # This prepares data for fully connected layers
        layers.Flatten(),
        
        # ============================================
        # FULLY CONNECTED LAYERS (Decision Making)
        # ============================================
        # These layers make the final classification decision
        
        # Dense: Fully connected layer (every neuron connected to every input)
        # 512 neurons: Number of decision-making units
        # activation='relu': ReLU activation
        layers.Dense(512, activation='relu'),
        
        # Dropout: Randomly turns off 50% of neurons during training
        # Prevents overfitting (memorizing training data too well)
        # rate=0.5: 50% of neurons are dropped
        layers.Dropout(0.5),
        
        # Another dense layer
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # ============================================
        # OUTPUT LAYER
        # ============================================
        # Final layer: Makes the prediction
        # For binary classification: 1 output neuron (probability of dog)
        # activation='sigmoid': Outputs probability between 0 and 1
        #   0 = cat, 1 = dog
        layers.Dense(1, activation='sigmoid')  # Changed to 1 output for binary classification
    ])
    
    # ============================================
    # COMPILE THE MODEL
    # ============================================
    # This configures the model for training
    
    model.compile(
        # Optimizer: Algorithm that updates model weights during training
        # 'adam': Adaptive Moment Estimation (popular, works well)
        # learning_rate: How big steps to take when learning (0.001 = small steps)
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        
        # Loss function: Measures how wrong the predictions are
        # 'binary_crossentropy': Good for binary classification (2 classes)
        # The model tries to minimize this value
        loss='binary_crossentropy',
        
        # Metrics: What to track during training
        # 'accuracy': Percentage of correct predictions
        metrics=['accuracy']
    )
    
    # Print model summary
    # Shows architecture, number of parameters, etc.
    print("=" * 50)
    print("Model Architecture:")
    print("=" * 50)
    model.summary()
    
    return model


# ============================================
# FUNCTION: Create Advanced Model (Optional)
# ============================================
def create_advanced_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a more advanced CNN with more layers.
    Use this for better accuracy (but slower training).
    """
    
    model = models.Sequential([
        # More convolutional blocks for better feature detection
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(1, activation='sigmoid')  # Binary classification: 1 output
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("=" * 50)
    print("Advanced Model Architecture:")
    print("=" * 50)
    model.summary()
    
    return model


# ============================================
# FUNCTION: Create Transfer Learning Model (Best Accuracy)
# ============================================
def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a model using transfer learning.
    
    Transfer Learning Explanation:
    - Uses a pre-trained model (trained on millions of images)
    - We add our own layers on top
    - Much faster training and better accuracy
    - Uses MobileNetV2 (lightweight, fast)
    """
    
    # Load pre-trained MobileNetV2 model
    # include_top=False: Don't include the final classification layers
    # weights='imagenet': Use weights trained on ImageNet dataset
    # input_shape: Size of input images
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers (don't retrain them)
    # This speeds up training
    base_model.trainable = False
    
    # Create new model with base + our layers
    model = models.Sequential([
        base_model,  # Pre-trained feature extractor
        
        # Global Average Pooling: Reduces dimensions
        layers.GlobalAveragePooling2D(),
        
        # Our classification layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification: 1 output
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
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
    Test the model creation.
    Run: python src/model.py
    """
    
    print("Creating simple model...")
    model = create_simple_model()
    
    print("\n" + "=" * 50)
    print("Model created successfully!")
    print("=" * 50)
    print(f"Total parameters: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

