# ============================================
# UTILS.PY - Shared Utilities
# ============================================
# Common utilities used across the project to reduce code duplication
# ============================================

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Get project root directory (shared across all modules)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.
    
    Returns:
    -------
    str
        Absolute path to project root
    """
    return PROJECT_ROOT


def setup_paths() -> None:
    """
    Add project root to Python path for imports.
    """
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists, create if it doesn't.
    
    Parameters:
    ----------
    directory : str
        Directory path (can be relative or absolute)
    
    Returns:
    -------
    str
        Absolute path to the directory
    """
    if not os.path.isabs(directory):
        directory = os.path.join(PROJECT_ROOT, directory)
    
    os.makedirs(directory, exist_ok=True)
    return directory


def get_path(config_paths: dict, key: str, default: Optional[str] = None) -> str:
    """
    Get a path from config, resolving relative paths to absolute.
    
    Parameters:
    ----------
    config_paths : dict
        Paths section from config
    key : str
        Key to look up
    default : str, optional
        Default path if key not found
    
    Returns:
    -------
    str
        Absolute path
    """
    path = config_paths.get(key, default)
    if path is None:
        return None
    
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    
    return path


def create_results_structure(base_dir: Optional[str] = None) -> dict:
    """
    Create organized results directory structure for presentation.
    
    Parameters:
    ----------
    base_dir : str, optional
        Base directory for results (default: results/)
    
    Returns:
    -------
    dict
        Dictionary with paths to result subdirectories
    """
    if base_dir is None:
        base_dir = os.path.join(PROJECT_ROOT, 'results')
    
    base_dir = ensure_dir(base_dir)
    
    structure = {
        'base': base_dir,
        'models': ensure_dir(os.path.join(base_dir, 'models')),
        'plots': ensure_dir(os.path.join(base_dir, 'plots')),
        'reports': ensure_dir(os.path.join(base_dir, 'reports')),
        'predictions': ensure_dir(os.path.join(base_dir, 'predictions')),
        'validation': ensure_dir(os.path.join(base_dir, 'validation')),
        'comparison': ensure_dir(os.path.join(base_dir, 'comparison')),
    }
    
    return structure

