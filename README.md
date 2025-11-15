# Dogs vs Cats Image Classification Project

## Project Overview

This project builds a **deep learning model** to classify images as either **dogs** or **cats**. It uses **TensorFlow/Keras** to train neural networks with three different architectures for comparison.

---

## Project Structure

```
Dogs_vs_Cats/
├── README.md              # This file - complete documentation
├── requirements.txt       # Python packages needed
├── .gitignore            # Files excluded from Git
│
├── src/                  # Source code
│   ├── data_loader.py    # Load and preprocess images
│   ├── model.py          # Define neural network architectures
│   ├── train.py          # Train the model (MAIN SCRIPT)
│   ├── predict.py        # Make predictions on test images
│   ├── compare_models.py # Compare trained models
│   └── test_setup.py     # Test project setup
│
├── data/                 # Data files
│   ├── train/            # Training images (25,000 images)
│   ├── test/             # Test images (12,500 images)
│   └── sampleSubmission.csv
│
├── models/               # Saved trained models (created after training)
├── logs/                 # Training logs
└── venv/                 # Virtual environment
```

---

## Quick Start

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Install Dependencies (if needed)
```bash
pip install -r requirements.txt
```

### 3. Configure Training
Edit `src/train.py`:
- `MAX_TRAINING_IMAGES`: Number of images (None = all, or 1000, 5000 for testing)
- `EPOCHS`: Training rounds (10 recommended)
- `MODEL_TYPE`: `'simple'`, `'advanced'`, or `'transfer'`
- `BATCH_SIZE`: Images per batch (32 recommended)

### 4. Train Model
```bash
python src/train.py
```

### 5. Make Predictions
Edit `src/predict.py` with your model path, then:
```bash
python src/predict.py
```

### 6. Compare Models (After Training Multiple)
```bash
python src/compare_models.py
```

---

## What Happens After Training Completes?

### Step-by-Step Workflow:

**1. Training Completes**
- Model automatically saves to `models/dogs_vs_cats_transfer_final_TIMESTAMP.h5`
- You'll see: `TRAINING SUMMARY` with final accuracy
- Training time and model path displayed

**2. Compare All Three Models**
```bash
python src/compare_models.py
```
- Compares: Simple, Advanced, Transfer Learning models
- Creates visual charts (`model_comparison.png`)
- Generates detailed report (`model_comparison_report.txt`)
- Shows accuracy, loss, model size, training time

**3. Make Predictions on Test Images**
```bash
# Edit src/predict.py first - update MODEL_PATH to your best model
python src/predict.py
```
- Loads your trained model
- Predicts on all test images in `data/test/`
- Saves predictions to `predictions.csv`
- Format: `id,label` (0=cat, 1=dog)

**4. Analyze Results for Presentation**
- Review model comparison charts
- Note which model performed best
- Prepare to explain:
  - Why Transfer Learning is best (pre-trained knowledge)
  - Trade-offs: Speed vs Accuracy
  - Real-world applications

**5. Optional: Retrain with Full Dataset**
- If you want even better results:
  - Edit `src/train.py`: Set `MAX_TRAINING_IMAGES = None` (uses all 25K images)
  - Set `EPOCHS = 20` (more training rounds)
  - Re-run training (takes longer but better accuracy)

---

## Key Libraries

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computing (image arrays)
- **Matplotlib**: Visualization
- **Pandas**: Data manipulation (CSV files)
- **Pillow & OpenCV**: Image processing

---

## Model Architectures

### 1. Simple CNN
- **Complexity**: Low-Medium
- **Training Time**: 1-2 hours (5K images)
- **Expected Accuracy**: ~75-85%
- **Best For**: Learning basics, easy to explain

### 2. Advanced CNN
- **Complexity**: Medium-High
- **Training Time**: 1.5-2.5 hours (5K images)
- **Expected Accuracy**: ~80-88%
- **Best For**: Showing depth impact on performance

### 3. Transfer Learning (MobileNetV2)
- **Complexity**: Conceptually High (but efficient)
- **Training Time**: 1-1.5 hours (5K images) - FASTEST
- **Expected Accuracy**: ~85-92% - BEST
- **Best For**: Industry standard, best results

---

## Configuration Guide

### Training Settings (`src/train.py`)

```python
# Image settings
IMAGE_SIZE = (224, 224)  # Standard size for neural networks

# Training settings
BATCH_SIZE = 32          # Images per batch (16, 32, 64)
EPOCHS = 10              # Training rounds
VALIDATION_SPLIT = 0.2   # 20% for validation

# Model selection
MODEL_TYPE = 'simple'    # Options: 'simple', 'advanced', 'transfer'

# Data settings
MAX_TRAINING_IMAGES = 5000  # None = all 25K, or set number for testing
```

### For Presentation Comparison

Train all 3 models with same settings:
1. Set `MAX_TRAINING_IMAGES = 5000` (balanced speed/accuracy)
2. Train Simple: `MODEL_TYPE = 'simple'` → `python src/train.py`
3. Train Advanced: `MODEL_TYPE = 'advanced'` → `python src/train.py`
4. Train Transfer: `MODEL_TYPE = 'transfer'` → `python src/train.py`
5. Compare: `python src/compare_models.py`

---

## How to Monitor Training Status

### Quick Checks

**1. Is training running?**
```bash
ps aux | grep "python.*train" | grep -v grep
```
- **Output shows process** = Training is running
- **No output** = Training stopped

**2. Check the log file:**
```bash
tail -f logs/training_transfer.log    # Follow live updates (Ctrl+C to exit)
tail -50 logs/training_transfer.log   # See last 50 lines
```

**3. Check log file size:**
```bash
ls -lh logs/training_transfer.log
```
- **File growing** = Training is writing logs
- **File size stays same** = Training might be stuck

### What to Look For in Logs

**Training is working correctly when you see:**
```
STEP 1: Loading images...
  Loading images from: data/train
  Loaded 5000 images

STEP 2: Splitting data...
  Training: 4000 images
  Validation: 1000 images

STEP 3: Creating neural network model...
✓ Model created

STEP 4: Setting up training callbacks...
✓ Callbacks configured

STEP 5: Training the model...
Epoch 1/10
  1/125 [====>] - 45s 360ms/step - accuracy: 0.5312 - loss: 0.6931
  2/125 [====>] - 44s 352ms/step - accuracy: 0.5625 - loss: 0.6754
  ...
Epoch 1: val_accuracy improved from -inf to 0.85200
```

**Problems to watch for:**
- `ModuleNotFoundError` = Missing packages
- `FileNotFoundError` = Wrong data path
- `OutOfMemoryError` = Too many images/batch size too large
- `KeyboardInterrupt` = You stopped it (Ctrl+C)

### Real-Time Monitoring

**Watch training progress live:**
```bash
# Follow log file updates (updates automatically)
tail -f logs/training_transfer.log

# Watch for specific info (accuracy, loss)
tail -f logs/training_transfer.log | grep -E "accuracy|loss|Epoch"
```

**Check which model files are being created:**
```bash
ls -lht models/*.h5 | head -5
```

### When Training Completes

Look for these messages:
```
TRAINING SUMMARY
Final accuracy: 85.20%
Final loss: 0.3245
Training time: 1h 23m 45s
Model saved to: models/dogs_vs_cats_transfer_final_20251115_153045.h5
```

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

## For Your Presentation

### What to Show:

1. **Three Different Approaches**
   - Simple CNN: Basic architecture
   - Advanced CNN: Deeper layers
   - Transfer Learning: Pre-trained model

2. **Comparison Metrics**
   - Training time per model
   - Final validation accuracy
   - Model complexity (parameters)
   - File sizes

3. **Implementation Differences**
   - Show code differences in `src/model.py`
   - Explain transfer learning concept
   - Compare training times

### Generate Comparison Report:
```bash
python src/compare_models.py
```
Creates:
- `model_comparison.png` - Visual charts
- `model_comparison_report.txt` - Detailed metrics

---

## Troubleshooting

**"ModuleNotFoundError"**
- Solution: Activate virtual environment first

**"No images loaded"**
- Solution: Check `data/train/` folder exists with `.jpg` files

**"Out of memory"**
- Solution: Reduce `BATCH_SIZE` to 16, or use fewer images

**"Training too slow"**
- Solution: Use Transfer Learning model (fastest), or reduce `MAX_TRAINING_IMAGES`

**"Model file not found"**
- Solution: Check `models/` folder, update `MODEL_PATH` in `src/predict.py`

---

## Common Commands

```bash
# Activate environment
source venv/bin/activate

# Test setup
python src/test_setup.py

# Test data loading
python src/data_loader.py

# Train model
python src/train.py

# Make predictions
python src/predict.py

# Compare models
python src/compare_models.py
```

---

## Understanding Training Output

```
Epoch 1/10
4000/4000 [==============================] - 120s 5ms/step - loss: 0.6931 - accuracy: 0.5000 - val_loss: 0.6931 - val_accuracy: 0.5500
```

- **Epoch**: Training round (1 of 10)
- **loss**: Error metric (lower is better)
- **accuracy**: Training accuracy (higher is better)
- **val_accuracy**: Validation accuracy (unseen data)

**Good signs**: Accuracy increasing, loss decreasing, validation close to training

---

## Important Notes

- **Large files excluded**: `data/train/`, `data/test/`, `models/*.h5` are not in Git
- **Always activate venv**: Before running any Python code
- **Training takes time**: Be patient, especially with full dataset
- **Models auto-save**: Best model saved automatically during training

---

## License

This is a learning project. Feel free to modify and experiment!
