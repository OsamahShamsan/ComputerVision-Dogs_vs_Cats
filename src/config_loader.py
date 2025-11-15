# ============================================
# CONFIG_LOADER.PY - Configuration File Loader
# ============================================
# This module provides utilities for loading and managing
# configuration files for different tasks (training, prediction, etc.)
# ============================================

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'configs')


def load_config(config_path: Optional[str] = None, config_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Parameters:
    ----------
    config_path : str, optional
        Full path to config file. If provided, this takes precedence.
    config_name : str, optional
        Name of config file (without extension) in configs/ directory.
        If config_path is not provided, loads from configs/{config_name}.yaml
    
    Returns:
    -------
    config : dict
        Configuration dictionary with all parameters
    
    Raises:
    ------
    FileNotFoundError
        If config file is not found
    ValueError
        If config file format is invalid
    """
    # Determine config file path
    if config_path:
        full_path = config_path
    elif config_name:
        # Try YAML first, then JSON
        yaml_path = os.path.join(CONFIG_DIR, f"{config_name}.yaml")
        json_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
        
        if os.path.exists(yaml_path):
            full_path = yaml_path
        elif os.path.exists(json_path):
            full_path = json_path
        else:
            raise FileNotFoundError(
                f"Config file not found: {yaml_path} or {json_path}\n"
                f"Available configs: {list_available_configs()}"
            )
    else:
        # Default to train.yaml
        default_path = os.path.join(CONFIG_DIR, 'train.yaml')
        if os.path.exists(default_path):
            full_path = default_path
        else:
            raise ValueError(
                "No config specified and default 'train.yaml' not found.\n"
                f"Please specify config_path or config_name.\n"
                f"Available configs: {list_available_configs()}"
            )
    
    # Load config file
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Config file not found: {full_path}")
    
    with open(full_path, 'r') as f:
        if full_path.endswith('.yaml') or full_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif full_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {full_path}")
    
    if config is None:
        raise ValueError(f"Config file is empty or invalid: {full_path}")
    
    # Resolve relative paths to absolute paths
    config = _resolve_paths(config, PROJECT_ROOT)
    
    return config


def _resolve_paths(config: Dict[str, Any], base_path: str) -> Dict[str, Any]:
    """
    Recursively resolve relative paths in config to absolute paths.
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary
    base_path : str
        Base path for resolving relative paths
    
    Returns:
    -------
    config : dict
        Configuration with resolved paths
    """
    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_paths(value, base_path)
        elif isinstance(value, str):
            # Check if it's a path-like string
            if any(keyword in key.lower() for keyword in ['path', 'dir', 'folder', 'file']):
                if not os.path.isabs(value):
                    resolved[key] = os.path.join(base_path, value)
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        else:
            resolved[key] = value
    return resolved


def list_available_configs() -> list:
    """
    List all available configuration files in the configs directory.
    
    Returns:
    -------
    configs : list
        List of available config file names (without extension)
    """
    if not os.path.exists(CONFIG_DIR):
        return []
    
    configs = set()
    for file in os.listdir(CONFIG_DIR):
        if file.endswith(('.yaml', '.yml', '.json')):
            configs.add(os.path.splitext(file)[0])
    
    return sorted(list(configs))


def save_config(config: Dict[str, Any], config_path: str, format: str = 'yaml'):
    """
    Save configuration dictionary to a file.
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary to save
    config_path : str
        Path where to save the config file
    format : str
        File format: 'yaml' or 'json' (default: 'yaml')
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        if format.lower() == 'yaml':
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif format.lower() == 'json':
            json.dump(config, f, indent=2, sort_keys=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Parameters:
    ----------
    base_config : dict
        Base configuration
    override_config : dict
        Configuration to override base values
    
    Returns:
    -------
    merged_config : dict
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def find_latest_model(models_dir, model_type=None):
    """
    Find the most recent model file in the models directory.
    
    This utility function is used by predict.py and compare_models.py to locate
    the most recently trained model, optionally filtered by model type.
    
    Parameters:
    ----------
    models_dir : str
        Path to models directory
    model_type : str, optional
        Filter by model type (e.g., 'transfer_learning', 'simple_cnn', 'advanced_cnn', 'deep_custom_cnn')
        Also supports old naming patterns: 'transfer', 'custom_cnn', 'deep_cnn'
    
    Returns:
    -------
    model_path : str or None
        Path to latest model file, or None if not found
    """
    import glob
    
    if not os.path.exists(models_dir):
        return None
    
    # Find all .h5 files
    pattern = os.path.join(models_dir, '*.h5')
    model_files = glob.glob(pattern)
    
    if model_type:
        model_files = [f for f in model_files if model_type in os.path.basename(f).lower()]
    
    if not model_files:
        return None
    
    # Sort by modification time (most recent first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]


# ============================================
# MAIN EXECUTION (for testing)
# ============================================
if __name__ == "__main__":
    """
    Test the config loader.
    Run: python src/config_loader.py
    """
    print("=" * 70)
    print("CONFIG LOADER TEST")
    print("=" * 70)
    print()
    
    print("Available config files:")
    configs = list_available_configs()
    if configs:
        for config in configs:
            print(f"  - {config}")
    else:
        print("  (No config files found in configs/ directory)")
    print()
    
    # Try loading a config if available
    if configs:
        try:
            config = load_config(config_name=configs[0])
            print(f"Successfully loaded config: {configs[0]}")
            print(f"Keys: {list(config.keys())}")
        except Exception as e:
            print(f"Error loading config: {e}")

