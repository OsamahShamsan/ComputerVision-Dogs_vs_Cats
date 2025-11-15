# ============================================
# COMPARE_MODELS.PY - Visual Comparison of Trained Models
# ============================================
# This script compares different trained models and creates
# visualizations for your presentation
# ============================================

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Get project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================
# FUNCTION: Load Model Information
# ============================================
def get_model_info(model_path):
    """
    Get information about a trained model.
    
    Parameters:
    ----------
    model_path : str
        Path to .h5 model file
    
    Returns:
    -------
    info : dict
        Dictionary with model information
    """
    try:
        # Load model
        model = keras.models.load_model(model_path)
        
        # Get model name from filename
        model_name = os.path.basename(model_path).replace('.h5', '')
        
        # Extract model type from filename
        if 'simple' in model_name.lower():
            model_type = 'Simple CNN'
        elif 'advanced' in model_name.lower():
            model_type = 'Advanced CNN'
        elif 'transfer' in model_name.lower():
            model_type = 'Transfer Learning'
        else:
            model_type = 'Unknown'
        
        # Get file size
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Count parameters
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        info = {
            'name': model_name,
            'type': model_type,
            'file_path': model_path,
            'file_size_mb': file_size_mb,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'layers': len(model.layers),
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape)
        }
        
        return info
        
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

# ============================================
# FUNCTION: Find All Trained Models
# ============================================
def find_all_models(models_dir=None):
    """
    Find all .h5 model files in the models directory.
    
    Returns:
    -------
    model_files : list
        List of model file paths
    """
    if models_dir is None:
        models_dir = os.path.join(PROJECT_ROOT, 'models')
    if not os.path.exists(models_dir):
        print(f"Error: '{models_dir}' directory not found!")
        return []
    
    # Find all .h5 files
    model_files = glob.glob(os.path.join(models_dir, '*.h5'))
    
    if not model_files:
        print(f"No model files found in '{models_dir}' directory.")
        print("Please train some models first using: python src/train.py")
        return []
    
    return sorted(model_files)

# ============================================
# FUNCTION: Create Comparison Visualization
# ============================================
def create_comparison_plot(model_infos, save_path=None):
    """
    Create a visualization comparing different models.
    
    Parameters:
    ----------
    model_infos : list
        List of model info dictionaries
    save_path : str
        Path to save the plot
    """
    if save_path is None:
        save_path = os.path.join(PROJECT_ROOT, 'logs', 'model_comparison.png')
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not model_infos:
        print("No model information to plot!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison for Dogs vs Cats Classification', 
                 fontsize=16, fontweight='bold')
    
    # Extract data
    model_names = [info['type'] for info in model_infos]
    param_counts = [info['total_parameters'] / 1e6 for info in model_infos]  # In millions
    file_sizes = [info['file_size_mb'] for info in model_infos]
    layer_counts = [info['layers'] for info in model_infos]
    trainable_params = [info['trainable_parameters'] / 1e6 for info in model_infos]
    non_trainable_params = [info['non_trainable_parameters'] / 1e6 for info in model_infos]
    
    # Plot 1: Total Parameters
    ax1 = axes[0, 0]
    bars1 = ax1.bar(model_names, param_counts, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_ylabel('Parameters (Millions)', fontsize=11)
    ax1.set_title('Model Complexity: Total Parameters', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}M', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Model File Size
    ax2 = axes[0, 1]
    bars2 = ax2.bar(model_names, file_sizes, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax2.set_ylabel('File Size (MB)', fontsize=11)
    ax2.set_title('Model Storage Size', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}MB', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Number of Layers
    ax3 = axes[1, 0]
    bars3 = ax3.bar(model_names, layer_counts, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax3.set_ylabel('Number of Layers', fontsize=11)
    ax3.set_title('Model Architecture: Layer Count', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Trainable vs Non-trainable Parameters
    ax4 = axes[1, 1]
    x_pos = np.arange(len(model_names))
    width = 0.35
    bars4a = ax4.bar(x_pos - width/2, trainable_params, width, 
                     label='Trainable', color='#3498db')
    bars4b = ax4.bar(x_pos + width/2, non_trainable_params, width,
                     label='Non-trainable', color='#95a5a6')
    ax4.set_ylabel('Parameters (Millions)', fontsize=11)
    ax4.set_title('Trainable vs Non-trainable Parameters', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(model_names)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {save_path}")
    plt.close()

# ============================================
# FUNCTION: Print Comparison Table
# ============================================
def print_comparison_table(model_infos):
    """
    Print a formatted comparison table.
    """
    print("\n" + "=" * 90)
    print("MODEL COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Model Type':<20} {'Parameters':<15} {'File Size':<12} {'Layers':<8} {'Trainable':<15}")
    print("-" * 90)
    
    for info in model_infos:
        params_str = f"{info['total_parameters']/1e6:.2f}M"
        size_str = f"{info['file_size_mb']:.1f}MB"
        layers_str = str(info['layers'])
        trainable_str = f"{info['trainable_parameters']/1e6:.2f}M"
        
        print(f"{info['type']:<20} {params_str:<15} {size_str:<12} {layers_str:<8} {trainable_str:<15}")
    
    print("=" * 90)

# ============================================
# FUNCTION: Generate Report
# ============================================
def generate_report(model_infos, report_path=None):
    """
    Generate a detailed text report.
    """
    if report_path is None:
        report_path = os.path.join(PROJECT_ROOT, 'logs', 'model_comparison_report.txt')
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("MODEL COMPARISON REPORT - DOGS VS CATS CLASSIFICATION\n")
        f.write("=" * 90 + "\n\n")
        
        for i, info in enumerate(model_infos, 1):
            f.write(f"MODEL {i}: {info['type']}\n")
            f.write("-" * 90 + "\n")
            f.write(f"  Name: {info['name']}\n")
            f.write(f"  File: {info['file_path']}\n")
            f.write(f"  File Size: {info['file_size_mb']:.2f} MB\n")
            f.write(f"  Total Parameters: {info['total_parameters']:,} ({info['total_parameters']/1e6:.2f}M)\n")
            f.write(f"  Trainable Parameters: {info['trainable_parameters']:,} ({info['trainable_parameters']/1e6:.2f}M)\n")
            f.write(f"  Non-trainable Parameters: {info['non_trainable_parameters']:,}\n")
            f.write(f"  Number of Layers: {info['layers']}\n")
            f.write(f"  Input Shape: {info['input_shape']}\n")
            f.write(f"  Output Shape: {info['output_shape']}\n")
            f.write("\n")
        
        f.write("=" * 90 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 90 + "\n")
        f.write("\nComplexity Comparison:\n")
        f.write("  - Simple CNN: Basic architecture, moderate complexity\n")
        f.write("  - Advanced CNN: Deeper layers, higher complexity\n")
        f.write("  - Transfer Learning: Pre-trained base, efficient fine-tuning\n")
        f.write("\nUse Cases:\n")
        f.write("  - Simple CNN: Good for learning, moderate accuracy\n")
        f.write("  - Advanced CNN: Better accuracy, requires more training time\n")
        f.write("  - Transfer Learning: Best accuracy, fastest training, industry standard\n")
    
    print(f"✓ Detailed report saved to: {report_path}")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    """
    Compare all trained models.
    
    Usage:
        python compare_models.py
    """
    
    print("=" * 70)
    print("MODEL COMPARISON TOOL")
    print("=" * 70)
    print("\nFinding trained models...")
    
    # Find all models
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    model_files = find_all_models(models_dir)
    
    if not model_files:
        print("\nNo models found. Please train some models first:")
        print("  python src/train.py")
        print("  OR")
        print("  python run_all_experiments.py")
        exit(1)
    
    print(f"✓ Found {len(model_files)} model(s)")
    
    # Load information for each model
    print("\nLoading model information...")
    model_infos = []
    for model_file in model_files:
        info = get_model_info(model_file)
        if info:
            model_infos.append(info)
            print(f"  ✓ Loaded: {info['type']}")
    
    if not model_infos:
        print("No valid models could be loaded!")
        exit(1)
    
    # Print comparison table
    print_comparison_table(model_infos)
    
    # Generate visualizations
    print("\nGenerating comparison visualizations...")
    create_comparison_plot(model_infos)
    
    # Generate report
    print("\nGenerating detailed report...")
    generate_report(model_infos)
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - logs/model_comparison.png (visual comparison)")
    print("  - logs/model_comparison_report.txt (detailed report)")
    print("\nThese files are perfect for your presentation!")
    print()

