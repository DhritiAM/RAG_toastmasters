# utils/paths.py
from pathlib import Path

# REPO_ROOT is the root directory of the project
# This file is at utils/paths.py, so parent.parent gives us the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

def resolve_path(path_str):
    """
    Resolve a relative path string relative to REPO_ROOT.
    
    Args:
        path_str: Relative path string (e.g., "data/vectordb/vector_index.faiss")
    
    Returns:
        Resolved absolute Path object
    """
    if isinstance(path_str, Path):
        # If already a Path, resolve relative to REPO_ROOT if it's relative
        if path_str.is_absolute():
            return path_str.resolve()
        return (REPO_ROOT / path_str).resolve()
    return (REPO_ROOT / path_str).resolve()
