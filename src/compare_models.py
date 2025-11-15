# ============================================
# COMPARE_MODELS.PY - Visual Comparison of Trained Models
# ============================================
# This script compares different trained models and creates
# visualizations and reports for analysis.
# It supports configuration via YAML files for flexible usage.
#
# Usage:
#   python src/compare_models.py                    # Uses default config (compare.yaml)
#   python src/compare_models.py --config compare   # Uses configs/compare.yaml
#   python src/compare_models.py --config-path custom.yaml
# ============================================

import os
import sys
import argparse
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Setup paths and import shared utilities
from src.utils import setup_paths, get_project_root, ensure_dir, get_path, create_results_structure
setup_paths()

from src.config_loader import load_config, find_latest_model

PROJECT_ROOT = get_project_root()

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
        # Checks for new model type names first, then falls back to old names for backward compatibility
        if 'simple_cnn' in model_name.lower() or 'simple' in model_name.lower():
            model_type = 'Simple CNN'
        elif 'advanced_cnn' in model_name.lower() or 'advanced' in model_name.lower():
            model_type = 'Advanced CNN'
        elif 'deep_custom_cnn' in model_name.lower() or 'deep_custom' in model_name.lower():
            model_type = 'Deep Custom CNN'
        elif 'transfer_learning' in model_name.lower() or 'transfer' in model_name.lower():
            model_type = 'Transfer Learning'
        # Backward compatibility with old naming
        elif 'custom_cnn' in model_name.lower():
            model_type = 'Simple CNN'  # Map old name to new name
        elif 'deep_cnn' in model_name.lower() and 'custom' not in model_name.lower():
            model_type = 'Advanced CNN'  # Map old name to new name
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
def create_comparison_plot(model_infos, save_path=None, config=None):
    """
    Create a visualization comparing different models.
    
    Parameters:
    ----------
    model_infos : list
        List of model info dictionaries
    save_path : str
        Path to save the plot
    config : dict, optional
        Configuration dictionary for visualization settings
    """
    if config is None:
        config = {}
    
    if save_path is None:
        save_path = os.path.join(PROJECT_ROOT, 'results', 'comparison', 'model_comparison.png')
    
    # Ensure directory exists
    ensure_dir(os.path.dirname(save_path))
    
    # Get visualization settings from config
    vis_config = config.get('visualization', {})
    fig_size = vis_config.get('figure_size', [14, 10])
    dpi = config.get('output', {}).get('plot_dpi', 300)
    if not model_infos:
        print("No model information to plot!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
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
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
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
        report_path = os.path.join(PROJECT_ROOT, 'results', 'comparison', 'model_comparison_report.txt')
    # Ensure directory exists
    ensure_dir(os.path.dirname(report_path))
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
        f.write("  - Simple CNN: 3 convolutional blocks, moderate complexity, trains from scratch\n")
        f.write("  - Advanced CNN: 4 convolutional blocks, higher complexity, trains from scratch\n")
        f.write("  - Deep Custom CNN: 5+ convolutional blocks, maximum depth, trains from scratch\n")
        f.write("  - Transfer Learning: Pre-trained base model (MobileNetV2), efficient fine-tuning\n")
        f.write("\nUse Cases:\n")
        f.write("  - Simple CNN: Good for learning, moderate accuracy (~75-85%), faster training\n")
        f.write("  - Advanced CNN: Better accuracy (~80-88%), requires more training time\n")
        f.write("  - Deep Custom CNN: High accuracy potential (~82-90%), significant training time\n")
        f.write("  - Transfer Learning: Best accuracy (~85-92%), fastest training, industry standard\n")
    
    print(f"Detailed report saved to: {report_path}")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    """
    Main entry point for model comparison script.
    Supports command-line arguments for configuration selection.
    """
    parser = argparse.ArgumentParser(
        description='Compare trained models and generate visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/compare_models.py                    # Use default config (compare.yaml)
  python src/compare_models.py --config compare   # Use configs/compare.yaml
  python src/compare_models.py --config-path custom.yaml  # Use custom config file
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='compare',
        help='Name of config file in configs/ directory (without extension)'
    )
    
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Full path to custom configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(config_path=args.config_path, config_name=args.config if not args.config_path else None)
    
    # Extract configuration sections
    paths = config.get('paths', {})
    models_config = config.get('models', {})
    output_config = config.get('output', {})
    
    print("=" * 70)
    print("MODEL COMPARISON TOOL")
    print("=" * 70)
    print("\nFinding trained models...")
    
    # Create results structure
    results_dir = create_results_structure()
    
    # Find models based on configuration
    models_dir = get_path(paths, 'models_dir', os.path.join(PROJECT_ROOT, 'models'))
    selection = models_config.get('selection', 'all')
    
    if selection == 'all':
        model_files = find_all_models(models_dir)
    elif selection == 'latest':
        # Find latest of each type
        # Searches for the most recent model of each architecture type
        model_files = []
        for model_type in ['simple_cnn', 'advanced_cnn', 'deep_custom_cnn', 'transfer_learning']:
            latest = find_latest_model(models_dir, model_type)
            if latest:
                model_files.append(latest)
        # Also check for old naming patterns for backward compatibility
        for old_name, new_name in [('custom_cnn', 'simple_cnn'), ('deep_cnn', 'advanced_cnn'), ('transfer', 'transfer_learning')]:
            if not any(new_name in f.lower() for f in model_files):
                latest = find_latest_model(models_dir, old_name)
                if latest:
                    model_files.append(latest)
    elif selection == 'specific':
        specific_models = models_config.get('specific_models', [])
        model_files = [os.path.join(models_dir, m) for m in specific_models if os.path.exists(os.path.join(models_dir, m))]
    else:
        model_files = find_all_models(models_dir)
    
    if not model_files:
        print("\nNo models found. Please train some models first:")
        print("  python src/train.py")
        exit(1)
    
    print(f"Found {len(model_files)} model(s)")
    
    # Load information for each model
    print("\nLoading model information...")
    model_infos = []
    for model_file in model_files:
        info = get_model_info(model_file)
        if info:
            model_infos.append(info)
            print(f"  Loaded: {info['type']}")
    
    if not model_infos:
        print("No valid models could be loaded!")
        exit(1)
    
    # Print comparison table
    print_comparison_table(model_infos)
    
    # Generate visualizations
    if output_config.get('comparison_plot'):
        print("\nGenerating comparison visualizations...")
        plot_path = output_config.get('comparison_plot')
        if not os.path.isabs(plot_path):
            plot_path = os.path.join(results_dir['comparison'], os.path.basename(plot_path))
        ensure_dir(os.path.dirname(plot_path))
        create_comparison_plot(model_infos, save_path=plot_path, config=config)
    
    # Generate report
    if output_config.get('report_file'):
        print("\nGenerating detailed report...")
        report_path = output_config.get('report_file')
        if not os.path.isabs(report_path):
            report_path = os.path.join(results_dir['comparison'], os.path.basename(report_path))
        ensure_dir(os.path.dirname(report_path))
        generate_report(model_infos, report_path=report_path)
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    if output_config.get('comparison_plot'):
        print(f"  - {output_config.get('comparison_plot')} (visual comparison)")
    if output_config.get('report_file'):
        print(f"  - {output_config.get('report_file')} (detailed report)")
    print()

