# ============================================
# __INIT__.PY - Package Initialization
# ============================================
# This file makes 'src' a Python package and exports common utilities
# ============================================

# Export common utilities
from src.utils import (
    get_project_root,
    setup_paths,
    ensure_dir,
    get_path,
    create_results_structure,
    PROJECT_ROOT
)

# Export config loader utilities
from src.config_loader import (
    load_config,
    find_latest_model,
    list_available_configs,
    save_config,
    merge_configs
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Dogs vs Cats Classification Project"

# Initialize paths on import
setup_paths()

__all__ = [
    # Utils
    'get_project_root',
    'setup_paths',
    'ensure_dir',
    'get_path',
    'create_results_structure',
    'PROJECT_ROOT',
    # Config
    'load_config',
    'find_latest_model',
    'list_available_configs',
    'save_config',
    'merge_configs',
    # Metadata
    '__version__',
    '__author__',
]

