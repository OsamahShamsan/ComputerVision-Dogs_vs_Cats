# Dogs vs Cats Image Classification Project

## Project Overview

This project implements a deep learning model to classify images as either dogs or cats. The system uses TensorFlow/Keras to train neural networks with four different architectures for comparison and analysis.

---

## Project Structure

```
Dogs_vs_Cats/
├── README.md              # Complete documentation
├── requirements.txt       # Python package dependencies
├── setup.py              # Package installation script
├── LICENSE               # MIT License
├── .gitignore            # Files excluded from Git
├── .venv_activate.sh     # Virtual environment activation script
│
├── src/                  # Source code
│   ├── __init__.py       # Package initialization
│   ├── config_loader.py  # Configuration file loader (YAML/JSON)
│   ├── data_loader.py    # Image loading and preprocessing
│   ├── model.py          # Neural network architectures
│   ├── train.py          # Model training script
│   ├── predict.py        # Prediction script
│   ├── validate.py       # Model validation and evaluation
│   ├── compare_models.py # Model comparison tool
│   ├── utils.py          # Shared utilities
│   └── test_setup.py     # Setup verification script
│
├── configs/              # Configuration files (YAML)
│   ├── base.yaml         # Base configuration (shared settings)
│   ├── train.yaml        # Training configuration
│   ├── predict.yaml      # Prediction configuration
│   ├── compare.yaml      # Model comparison configuration
│   ├── debug.yaml        # Debug/testing configuration
│   └── validate.yaml     # Validation configuration
│
├── data/                 # Data files
│   ├── train/            # Training images (25,000 images)
│   ├── test/             # Test images (12,500 images)
│   └── sampleSubmission.csv
│
├── models/               # Saved trained models
├── logs/                 # Training logs
├── results/              # Organized results for presentation
│   ├── models/           # Best trained models (copies)
│   ├── plots/            # Visualization plots
│   ├── reports/          # Text reports
│   ├── predictions/      # Prediction outputs
│   ├── validation/       # Validation results
│   └── comparison/       # Model comparison results
└── venv/                 # Virtual environment
```

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Sufficient disk space (at least 10GB recommended for full dataset)

### Step 1: Create Virtual Environment

A virtual environment isolates project dependencies from system Python and other projects, ensuring consistency and reproducibility.

```bash
python3 -m venv venv
```

### Step 2: Activate Virtual Environment

**Option 1: Using activation script (recommended)**
```bash
source .venv_activate.sh
```

The activation script verifies the virtual environment is properly activated and displays Python version information.

**Option 2: Manual activation**
```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

**Verification:**
After activation, the terminal prompt should show `(venv)`. Verification commands:
```bash
which python  # Should show path to venv/bin/python
echo $VIRTUAL_ENV  # Should show path to venv directory
```

**Important:** The virtual environment must be activated before running any Python commands or scripts.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Setup

```bash
python src/test_setup.py
```

This script checks:
- Data directories exist
- Required Python packages are installed
- Source files are present
- Sufficient disk space available
- GPU availability (if applicable)

---

## Quick Start

### Training a Model

**Using default configuration:**
```bash
python src/train.py
```

**Using specific configuration:**
```bash
python src/train.py --config train      # Default training config
python src/train.py --config debug      # Quick debug config (faster)
python src/train.py --config-path path/to/custom.yaml  # Custom config
```

### Making Predictions

**Using default configuration:**
```bash
python src/predict.py
```

The configuration automatically finds the latest trained model. To specify a model, edit `configs/predict.yaml`.

**Using specific configuration:**
```bash
python src/predict.py --config predict
python src/predict.py --config-path path/to/custom.yaml
```

### Validating a Model

**Using default configuration:**
```bash
python src/validate.py
```

This generates validation metrics, confusion matrix, and sample prediction visualizations.

**Using specific configuration:**
```bash
python src/validate.py --config validate
python src/validate.py --config-path path/to/custom.yaml
```

### Comparing Models

After training multiple models:
```bash
python src/compare_models.py
```

This creates comparison visualizations and detailed reports in `results/comparison/`.

---

## Configuration System

The project uses YAML configuration files to manage all parameters, enabling experimentation without modifying source code.

### Configuration Files

All configuration files are located in the `configs/` directory:

- **base.yaml** - Base configuration with shared settings
- **train.yaml** - Default training configuration
- **predict.yaml** - Default prediction configuration
- **compare.yaml** - Model comparison configuration
- **debug.yaml** - Quick debugging configuration (small dataset, few epochs)
- **validate.yaml** - Model validation configuration

### Configuration Structure

#### Training Configuration (train.yaml)

```yaml
# Image preprocessing
image:
  size: [224, 224]  # Image dimensions
  channels: 3       # RGB

# Data settings
data:
  max_training_images: 5000  # null for full dataset (all 25K images)
  validation_split: 0.2        # 20% for validation
  random_seed: 42             # For reproducibility

# Model architecture
model:
  type: "simple_cnn"  # Options: "simple_cnn", "advanced_cnn", "deep_custom_cnn", "transfer_learning"
  simple_cnn:
    conv_blocks: 3
    filters: [32, 64, 128]
    dense_units: [512, 256]
    dropout_rate: 0.5
    learning_rate: 0.001

# Training hyperparameters
training:
  batch_size: 32
  epochs: 10
  optimizer: "adam"
  loss: "binary_crossentropy"
```

#### Prediction Configuration (predict.yaml)

```yaml
# Model selection
model:
  path: "latest"  # "latest" or specific model path

# Image preprocessing
image:
  size: [224, 224]  # Must match training size

# Prediction settings
prediction:
  batch_size: 32
  output_csv: "results/predictions/predictions.csv"
  threshold: 0.5
```

### Creating Custom Configurations

1. Copy an existing config file:
   ```bash
   cp configs/train.yaml configs/my_custom_config.yaml
   ```

2. Edit parameters as needed

3. Use the custom config:
   ```bash
   python src/train.py --config my_custom_config
   ```

### Common Configuration Scenarios

**Quick Debugging:**
```bash
python src/train.py --config debug
```
Uses smaller images (128x128), fewer images (100), and fewer epochs (2).

**Full Dataset Training:**
Edit `configs/train.yaml`:
```yaml
data:
  max_training_images: null  # Use all images
training:
  epochs: 20  # More epochs
```

**Different Model Architecture:**
Edit `configs/train.yaml`:
```yaml
model:
  type: "transfer_learning"  # Use transfer learning
  transfer_learning:
    base_model: "ResNet50"  # Different base model
```

### Benefits of Configuration Files

- **No Code Changes:** Adjust all parameters via config files
- **Reproducibility:** Share config files to reproduce experiments
- **Version Control:** Track config changes in git
- **Flexibility:** Easy to create task-specific configs
- **Documentation:** Config files serve as self-documenting parameter lists

---

## Model Architectures

The project supports four neural network architectures, each with different complexity, training time, and accuracy characteristics.

### 1. Simple CNN (simple_cnn)

- **Complexity:** Low-Medium
- **Architecture:** 3 convolutional blocks, trains from scratch
- **Training Time:** 1-2 hours (5K images)
- **Expected Accuracy:** ~75-85%
- **Use Cases:** Learning basics, faster training, moderate accuracy
- **Parameters:** Moderate number of trainable parameters

### 2. Advanced CNN (advanced_cnn)

- **Complexity:** Medium-High
- **Architecture:** 4 convolutional blocks, trains from scratch
- **Training Time:** 1.5-2.5 hours (5K images)
- **Expected Accuracy:** ~80-88%
- **Use Cases:** Better accuracy, demonstrates depth impact on performance
- **Parameters:** More parameters than simple CNN

### 3. Deep Custom CNN (deep_custom_cnn)

- **Complexity:** High
- **Architecture:** 5+ convolutional blocks, trains from scratch
- **Training Time:** 2-3.5 hours (5K images)
- **Expected Accuracy:** ~82-90%
- **Use Cases:** Maximum accuracy from scratch training, deep architecture
- **Parameters:** Most parameters among from-scratch models

### 4. Transfer Learning (transfer_learning)

- **Complexity:** Conceptually High (but efficient)
- **Architecture:** Pre-trained base model (MobileNetV2) with custom layers
- **Training Time:** 1-1.5 hours (5K images) - Fastest
- **Expected Accuracy:** ~85-92% - Best
- **Use Cases:** Industry standard, best results, fastest training
- **Parameters:** Many pre-trained (frozen) parameters, fewer trainable parameters

---

## Workflow After Training

### Step 1: Training Completion

When training completes:
- Model automatically saves to `models/dogs_vs_cats_{model_type}_final_TIMESTAMP.h5`
- A copy is saved to `results/models/` for presentation
- Training summary displays final accuracy and training time
- Model path is displayed

### Step 2: Model Comparison

```bash
python src/compare_models.py
```

This generates:
- Comparison visualizations in `results/comparison/model_comparison.png`
- Detailed report in `results/comparison/model_comparison_report.txt`
- Metrics include: accuracy, loss, model size, training time

### Step 3: Making Predictions

```bash
python src/predict.py
```

This process:
- Loads the trained model (automatically finds latest or uses config)
- Predicts on all test images in `data/test/`
- Saves predictions to `results/predictions/predictions.csv`
- Format: `id,label` (0=cat, 1=dog)

### Step 4: Model Validation

```bash
python src/validate.py
```

This generates:
- Validation metrics (accuracy, precision, recall, F1-score)
- Confusion matrix plot in `results/validation/confusion_matrix.png`
- Sample prediction visualizations in `results/validation/prediction_samples.png`
- Detailed results report in `results/validation/validation_results.txt`

### Step 5: Full Dataset Training (Optional)

For improved results:
1. Edit `configs/train.yaml`: Set `data.max_training_images: null` (uses all 25K images)
2. Set `training.epochs: 20` (more training rounds)
3. Re-run training (takes longer but improves accuracy)
4. Results are automatically organized in `results/` folder

---

## Monitoring Training

### Checking Training Status

**Check if training is running:**
```bash
ps aux | grep "python.*train" | grep -v grep
```
- Output shows process: Training is running
- No output: Training stopped

**Monitor log file:**
```bash
tail -f logs/training.log    # Follow live updates (Ctrl+C to exit)
tail -50 logs/training.log   # See last 50 lines
```

**Check log file size:**
```bash
ls -lh logs/training.log
```
- File growing: Training is writing logs
- File size constant: Training might be stuck

### Training Output Indicators

**Normal training output:**
```
STEP 1: Loading images...
  Loading images from: data/train
  Loaded 5000 images

STEP 2: Splitting data...
  Training: 4000 images
  Validation: 1000 images

STEP 3: Creating neural network model...
  Model created

STEP 4: Setting up training callbacks...
  Callbacks configured

STEP 5: Training the model...
Epoch 1/10
  1/125 [====>] - 45s 360ms/step - accuracy: 0.5312 - loss: 0.6931
  2/125 [====>] - 44s 352ms/step - accuracy: 0.5625 - loss: 0.6754
  ...
Epoch 1: val_accuracy improved from -inf to 0.85200
```

**Problems to watch for:**
- `ModuleNotFoundError`: Missing packages
- `FileNotFoundError`: Wrong data path
- `OutOfMemoryError`: Too many images or batch size too large
- `KeyboardInterrupt`: Process stopped manually

### Understanding Training Metrics

```
Epoch 1/10
4000/4000 [==============================] - 120s 5ms/step - loss: 0.6931 - accuracy: 0.5000 - val_loss: 0.6931 - val_accuracy: 0.5500
```

- **Epoch:** Training round (1 of 10)
- **loss:** Error metric (lower is better)
- **accuracy:** Training accuracy (higher is better)
- **val_accuracy:** Validation accuracy on unseen data

**Good indicators:** Accuracy increasing, loss decreasing, validation accuracy close to training accuracy

---

## Performance Estimates

| Model | Images | Epochs | Time |
|-------|--------|--------|------|
| Simple | 500 | 3 | 15-30 min |
| Simple | 5,000 | 10 | 1-2 hours |
| Simple | 25,000 | 10 | 2-3 hours |
| Advanced | 5,000 | 10 | 1.5-2.5 hours |
| Transfer | 5,000 | 10 | 1-1.5 hours |

---

## Results Organization

All results are automatically organized in the `results/` directory:

- **results/models/** - Best trained models (copies from models/)
- **results/plots/** - Visualization plots and charts
- **results/reports/** - Text reports and summaries
- **results/predictions/** - Prediction outputs (CSV files)
- **results/validation/** - Model validation results
- **results/comparison/** - Model comparison visualizations and reports

This structure facilitates:
- Quick location of specific results
- Sharing results with others
- Including results in reports and presentations

---

## Virtual Environment Management

### Why Use a Virtual Environment?

A virtual environment isolates project Python packages from system Python and other projects, ensuring:
- **Consistency:** Same package versions across different machines
- **Isolation:** No conflicts with other projects
- **Reproducibility:** Easy to recreate the exact environment

### Activation

**Quick activation (recommended):**
```bash
source .venv_activate.sh
```

The script activates the virtual environment, verifies activation, and displays Python version and path.

**Manual activation:**
```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

**Verification:**
After activation, `(venv)` appears in the terminal prompt. Additional verification:
```bash
which python  # Should show path to venv/bin/python
echo $VIRTUAL_ENV  # Should show path to venv directory
```

### Deactivation

When finished working:
```bash
deactivate
```

### Installing Packages

Always activate the virtual environment first, then install:
```bash
source .venv_activate.sh  # or: source venv/bin/activate
pip install -r requirements.txt
```

### Recreating the Environment

If issues occur, the environment can be recreated:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Troubleshooting

### "ModuleNotFoundError"
**Solution:** Activate virtual environment first
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "No images loaded"
**Solution:** Verify `data/train/` folder exists with `.jpg` files
```bash
ls data/train/ | head -10
```

### "Out of memory"
**Solution:** Reduce batch size or use fewer images
- Edit `configs/train.yaml`: Set `training.batch_size: 16`
- Or reduce `data.max_training_images` to a smaller number

### "Training too slow"
**Solution:** Use Transfer Learning model (fastest) or reduce dataset size
- Edit `configs/train.yaml`: Set `model.type: "transfer_learning"`
- Or reduce `data.max_training_images` in config

### "Model file not found"
**Solution:** Check `models/` folder and update configuration
- Verify models exist: `ls models/*.h5`
- Edit `configs/predict.yaml`: Set `model.path: "latest"` for auto-detection
- Or specify exact model path in config

### "Command not found: python"
**Solution:** Ensure virtual environment is activated
- Check that `venv/bin/python` exists
- Activate venv: `source venv/bin/activate`

### "Package not found" errors
**Solution:** Activate venv and install packages
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## Common Commands

```bash
# Activate environment
source venv/bin/activate

# Verify setup
python src/test_setup.py

# Train model
python src/train.py

# Train with specific config
python src/train.py --config debug

# Make predictions
python src/predict.py

# Validate model
python src/validate.py

# Compare models
python src/compare_models.py

# Monitor training logs
tail -f logs/training.log
```

---

## Key Libraries

- **TensorFlow/Keras:** Deep learning framework
- **NumPy:** Numerical computing (image arrays)
- **Matplotlib:** Visualization
- **Pandas:** Data manipulation (CSV files)
- **Pillow & OpenCV:** Image processing
- **scikit-learn:** Machine learning utilities
- **PyYAML:** Configuration file parsing

---

## Important Notes

- **Large files excluded:** `data/train/`, `data/test/`, `models/*.h5` are not in Git
- **Always activate venv:** Required before running any Python code
- **Training takes time:** Be patient, especially with full dataset
- **Models auto-save:** Best model saved automatically during training
- **Results organization:** All outputs organized in `results/` folder for presentation

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Project Features

### Code Organization

- **Shared utilities:** Common functions in `src/utils.py` to reduce code duplication
- **Consistent paths:** All scripts use shared path utilities
- **Package structure:** Proper Python package with `__init__.py` exports
- **Configuration system:** YAML-based configuration for all parameters

### Results Management

- **Automatic organization:** Results saved to organized `results/` directory structure
- **Presentation ready:** All outputs formatted for easy inclusion in presentations
- **Version control friendly:** Results structure preserved in git while content is excluded

### Reproducibility

- **Configuration files:** All parameters in version-controlled YAML files
- **Random seeds:** Fixed random seeds for reproducible results
- **Environment isolation:** Virtual environment ensures consistent dependencies
