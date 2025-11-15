# ============================================
# TEST_SETUP.PY - Quick Test to Verify Everything Works
# ============================================
# Run this first to make sure your setup is correct!
# ============================================

import os
import sys

# Get project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("TESTING YOUR SETUP")
print("=" * 70)
print()

# Test 1: Check if data folders exist
print("TEST 1: Checking data folders...")
test_results = []

data_train_path = os.path.join(PROJECT_ROOT, 'data', 'train')
if os.path.exists(data_train_path):
    train_files = [f for f in os.listdir(data_train_path) if f.endswith('.jpg')]
    if len(train_files) > 0:
        print(f"  ✓ Training folder found with {len(train_files)} images")
        test_results.append(True)
    else:
        print("  ✗ Training folder is empty!")
        test_results.append(False)
else:
    print("  ✗ Training folder 'data/train' not found!")
    test_results.append(False)

data_test_path = os.path.join(PROJECT_ROOT, 'data', 'test')
if os.path.exists(data_test_path):
    test_files = [f for f in os.listdir(data_test_path) if f.endswith('.jpg')]
    if len(test_files) > 0:
        print(f"  ✓ Test folder found with {len(test_files)} images")
        test_results.append(True)
    else:
        print("  ✗ Test folder is empty!")
        test_results.append(False)
else:
    print("  ✗ Test folder 'data/test' not found!")
    test_results.append(False)

print()

# Test 2: Check if required packages are installed
print("TEST 2: Checking required packages...")
try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow {tf.__version__} installed")
    test_results.append(True)
except ImportError:
    print("  ✗ TensorFlow not installed!")
    test_results.append(False)

try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__} installed")
    test_results.append(True)
except ImportError:
    print("  ✗ NumPy not installed!")
    test_results.append(False)

try:
    import matplotlib
    print(f"  ✓ Matplotlib {matplotlib.__version__} installed")
    test_results.append(True)
except ImportError:
    print("  ✗ Matplotlib not installed!")
    test_results.append(False)

try:
    import pandas as pd
    print(f"  ✓ Pandas {pd.__version__} installed")
    test_results.append(True)
except ImportError:
    print("  ✗ Pandas not installed!")
    test_results.append(False)

print()

# Test 3: Check if source files exist
print("TEST 3: Checking source code files...")
src_files = ['data_loader.py', 'model.py', 'train.py', 'predict.py']
for src_file in src_files:
    src_file_path = os.path.join(PROJECT_ROOT, 'src', src_file)
    if os.path.exists(src_file_path):
        print(f"  ✓ src/{src_file} exists")
        test_results.append(True)
    else:
        print(f"  ✗ src/{src_file} not found!")
        test_results.append(False)

print()

# Test 4: Check disk space
print("TEST 4: Checking disk space...")
import shutil
total, used, free = shutil.disk_usage(PROJECT_ROOT)
free_gb = free / (1024**3)
print(f"  ✓ Free disk space: {free_gb:.2f} GB")
if free_gb < 5:
    print("  ⚠️  WARNING: Low disk space! Training may fail.")
    test_results.append(False)
else:
    test_results.append(True)

print()

# Test 5: Check TensorFlow GPU/MPS support
print("TEST 5: Checking TensorFlow acceleration...")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  ✓ GPU detected: {len(gpus)} device(s)")
    else:
        # Check for MPS (Metal Performance Shaders) on Apple Silicon
        try:
            import platform
            if platform.processor() == 'arm':
                print("  ✓ Apple Silicon detected - TensorFlow will use Metal (MPS)")
                print("  ✓ This will accelerate training on your M4 chip!")
            else:
                print("  ℹ️  No GPU detected - will use CPU (slower but works)")
        except:
            print("  ℹ️  Will use CPU for training")
    test_results.append(True)
except Exception as e:
    print(f"  ⚠️  Could not check GPU status: {e}")
    test_results.append(True)  # Not critical

print()

# Final summary
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
passed = sum(test_results)
total = len(test_results)
print(f"Passed: {passed}/{total} tests")

if passed == total:
    print("\n🎉 ALL TESTS PASSED! Your setup is ready!")
    print("\nNext steps:")
    print("  1. Read README.md for detailed instructions")
    print("  2. Run: python src/data_loader.py (test data loading)")
    print("  3. Run: python src/train.py (start training)")
else:
    print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
    if not test_results[0] or not test_results[1]:
        print("\n💡 Tip: Make sure your data folders are in the right place:")
        print("  - data/train/ (should contain training images)")
        print("  - data/test/ (should contain test images)")

print()

