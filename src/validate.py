# ============================================
# VALIDATE.PY - Model Validation and Evaluation
# ============================================
# This script validates trained models and generates evaluation metrics,
# confusion matrices, and visualization plots for presentation.
#
# Usage:
#   python src/validate.py                    # Uses default config (validate.yaml)
#   python src/validate.py --config validate  # Uses configs/validate.yaml
#   python src/validate.py --config-path custom.yaml
# ============================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import tensorflow as tf
from tensorflow import keras

# Setup paths
from src.utils import setup_paths, get_project_root, ensure_dir, get_path, create_results_structure
setup_paths()

from src.config_loader import load_config, find_latest_model
from src.data_loader import load_images_from_folder, split_data


def validate_model(config_path=None, config_name=None):
    """
    Validate a trained model and generate evaluation metrics.
    
    Parameters:
    ----------
    config_path : str, optional
        Full path to configuration file
    config_name : str, optional
        Name of configuration file in configs/ directory
    
    Returns:
    -------
    dict
        Dictionary containing validation results and metrics
    """
    # Load configuration
    config = load_config(config_path=config_path, config_name=config_name)
    
    # Extract configuration sections
    paths = config.get('paths', {})
    model_config = config.get('model', {})
    image_config = config.get('image', {})
    data_config = config.get('data', {})
    eval_config = config.get('evaluation', {})
    output_config = config.get('output', {})
    
    PROJECT_ROOT = get_project_root()
    
    # Create results structure
    results_dir = create_results_structure()
    
    print("=" * 70)
    print("MODEL VALIDATION AND EVALUATION")
    print("=" * 70)
    print()
    
    # ============================================
    # STEP 1: Load Model
    # ============================================
    print("STEP 1: Loading trained model...")
    print("-" * 70)
    
    models_dir = get_path(paths, 'models_dir', os.path.join(PROJECT_ROOT, 'models'))
    model_path_config = model_config.get('path', 'latest')
    
    if model_path_config == 'latest':
        model_path = find_latest_model(models_dir, model_config.get('model_type'))
        if model_path is None:
            print(f"ERROR: No model files found in '{models_dir}'")
            print("Please train a model first using: python src/train.py")
            return None
    else:
        model_path = model_path_config
        if not os.path.isabs(model_path):
            model_path = os.path.join(models_dir, model_path)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None
    
    model = keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    print(f"  Model type: {os.path.basename(model_path)}")
    print()
    
    # ============================================
    # STEP 2: Load Validation Data
    # ============================================
    print("STEP 2: Loading validation data...")
    print("-" * 70)
    
    train_dir = get_path(paths, 'train_dir', os.path.join(PROJECT_ROOT, 'data', 'train'))
    image_size = tuple(image_config.get('size', [224, 224]))
    max_images = data_config.get('max_images')
    
    # Load images
    images, labels, filenames = load_images_from_folder(
        folder_path=train_dir,
        target_size=image_size,
        max_images=max_images
    )
    
    if len(images) == 0:
        print(f"ERROR: No images loaded!")
        return None
    
    # Split data (using same split as training)
    validation_split = data_config.get('validation_split', 0.2)
    random_seed = data_config.get('random_seed', 42)
    
    _, X_val, _, y_val = split_data(
        images, labels,
        validation_split=validation_split,
        random_seed=random_seed
    )
    
    print(f"Loaded {len(X_val)} validation images")
    print(f"  - Cats: {np.sum(y_val == 0)}")
    print(f"  - Dogs: {np.sum(y_val == 1)}")
    print()
    
    # ============================================
    # STEP 3: Make Predictions
    # ============================================
    print("STEP 3: Making predictions on validation set...")
    print("-" * 70)
    
    batch_size = eval_config.get('batch_size', 32)
    predictions = model.predict(X_val, batch_size=batch_size, verbose=1)
    
    # Extract probabilities
    if len(predictions.shape) > 1 and predictions.shape[1] == 2:
        y_pred_proba = predictions[:, 1]
    else:
        y_pred_proba = predictions[:, 0] if len(predictions.shape) > 1 else predictions
    
    # Convert to binary predictions
    threshold = 0.5
    y_pred = (y_pred_proba > threshold).astype(int)
    
    print(f"Predictions complete")
    print()
    
    # ============================================
    # STEP 4: Calculate Metrics
    # ============================================
    print("STEP 4: Calculating evaluation metrics...")
    print("-" * 70)
    
    metrics = {}
    metrics_list = eval_config.get('metrics', ['accuracy'])
    
    if 'accuracy' in metrics_list:
        metrics['accuracy'] = accuracy_score(y_val, y_pred)
    if 'precision' in metrics_list:
        metrics['precision'] = precision_score(y_val, y_pred, zero_division=0)
    if 'recall' in metrics_list:
        metrics['recall'] = recall_score(y_val, y_pred, zero_division=0)
    if 'f1_score' in metrics_list or 'f1' in metrics_list:
        metrics['f1_score'] = f1_score(y_val, y_pred, zero_division=0)
    
    # Print metrics
    print("Validation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name.capitalize()}: {metric_value:.4f} ({metric_value*100:.2f}%)")
    print()
    
    # ============================================
    # STEP 5: Generate Confusion Matrix
    # ============================================
    results = {
        'model_path': model_path,
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'true_labels': y_val,
    }
    
    if eval_config.get('generate_confusion_matrix', True):
        print("STEP 5: Generating confusion matrix...")
        print("-" * 70)
        
        cm = confusion_matrix(y_val, y_pred)
        results['confusion_matrix'] = cm
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.colorbar()
        
        classes = ['Cat', 'Dog']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        cm_path = output_config.get('confusion_matrix_plot', 'results/validation/confusion_matrix.png')
        if not os.path.isabs(cm_path):
            cm_path = os.path.join(results_dir['validation'], os.path.basename(cm_path))
        ensure_dir(os.path.dirname(cm_path))
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {cm_path}")
        print(f"  True Negatives (Cat->Cat): {cm[0, 0]}")
        print(f"  False Positives (Cat->Dog): {cm[0, 1]}")
        print(f"  False Negatives (Dog->Cat): {cm[1, 0]}")
        print(f"  True Positives (Dog->Dog): {cm[1, 1]}")
        print()
    
    # ============================================
    # STEP 6: Generate Classification Report
    # ============================================
    if eval_config.get('generate_classification_report', True):
        print("STEP 6: Generating classification report...")
        print("-" * 70)
        
        report = classification_report(y_val, y_pred, target_names=['Cat', 'Dog'])
        results['classification_report'] = report
        
        print("Classification Report:")
        print(report)
        print()
    
    # ============================================
    # STEP 7: Visualize Sample Predictions
    # ============================================
    if eval_config.get('visualize_predictions', True):
        print("STEP 7: Creating prediction visualization...")
        print("-" * 70)
        
        num_samples = eval_config.get('num_samples_to_visualize', 16)
        indices = np.random.choice(len(X_val), min(num_samples, len(X_val)), replace=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Sample Validation Predictions', fontsize=14, fontweight='bold')
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(indices):
                i = indices[idx]
                img = X_val[i]
                true_label = 'Cat' if y_val[i] == 0 else 'Dog'
                pred_label = 'Cat' if y_pred[i] == 0 else 'Dog'
                confidence = y_pred_proba[i] if y_pred[i] == 1 else 1 - y_pred_proba[i]
                correct = 'CORRECT' if y_val[i] == y_pred[i] else 'WRONG'
                
                ax.imshow(img)
                ax.set_title(f'{correct} True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                           fontsize=9)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        vis_path = output_config.get('predictions_plot', 'results/validation/prediction_samples.png')
        if not os.path.isabs(vis_path):
            vis_path = os.path.join(results_dir['validation'], os.path.basename(vis_path))
        ensure_dir(os.path.dirname(vis_path))
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Prediction visualization saved to: {vis_path}")
        print()
    
    # ============================================
    # STEP 8: Save Results
    # ============================================
    print("STEP 8: Saving validation results...")
    print("-" * 70)
    
    # Save predictions CSV
    if output_config.get('save_predictions', True):
        pred_df = pd.DataFrame({
            'true_label': y_val,
            'predicted_label': y_pred,
            'probability': y_pred_proba
        })
        
        pred_path = output_config.get('predictions_file', 'results/validation/validation_predictions.csv')
        if not os.path.isabs(pred_path):
            pred_path = os.path.join(results_dir['validation'], os.path.basename(pred_path))
        ensure_dir(os.path.dirname(pred_path))
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions saved to: {pred_path}")
    
    # Save results report
    results_path = output_config.get('results_file', 'results/validation/validation_results.txt')
    if not os.path.isabs(results_path):
        results_path = os.path.join(results_dir['validation'], os.path.basename(results_path))
    ensure_dir(os.path.dirname(results_path))
    
    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL VALIDATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Validation Images: {len(X_val)}\n\n")
        f.write("METRICS\n")
        f.write("-" * 70 + "\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name.capitalize()}: {metric_value:.4f} ({metric_value*100:.2f}%)\n")
        f.write("\n")
        
        if 'confusion_matrix' in results:
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 70 + "\n")
            f.write(f"True Negatives (Cat->Cat): {cm[0, 0]}\n")
            f.write(f"False Positives (Cat->Dog): {cm[0, 1]}\n")
            f.write(f"False Negatives (Dog->Cat): {cm[1, 0]}\n")
            f.write(f"True Positives (Dog->Dog): {cm[1, 1]}\n\n")
        
        if 'classification_report' in results:
            f.write("CLASSIFICATION REPORT\n")
            f.write("-" * 70 + "\n")
            f.write(results['classification_report'])
    
    print(f"Results report saved to: {results_path}")
    print()
    
    # ============================================
    # SUMMARY
    # ============================================
    print("=" * 70)
    print("VALIDATION COMPLETE!")
    print("=" * 70)
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
    print(f"Results saved to: {results_dir['validation']}")
    print()
    
    return results


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Validate trained model and generate evaluation metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/validate.py                    # Use default config (validate.yaml)
  python src/validate.py --config validate  # Use configs/validate.yaml
  python src/validate.py --config-path custom.yaml  # Use custom config file
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='validate',
        help='Name of config file in configs/ directory (without extension)'
    )
    
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Full path to custom configuration file'
    )
    
    args = parser.parse_args()
    
    # Validate model
    results = validate_model(
        config_path=args.config_path,
        config_name=args.config if not args.config_path else None
    )
    
    if results is not None:
        print("Validation completed successfully")

