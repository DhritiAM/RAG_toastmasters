import json
import os
import re
from pathlib import Path

try:
    from ..utils.paths import REPO_ROOT, resolve_path
except ImportError:
    # Fallback for direct imports
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.paths import REPO_ROOT, resolve_path

class CoreLookup:
    """
    A lightweight lookup mechanism for broad knowledge questions.
    Loads pre-written canonical answers from core_knowledge.json.
    """

    def __init__(self, filepath=None):
        if filepath is None:
            filepath = str(REPO_ROOT / "data/static/core_knowledge.json")
        else:
            filepath = str(resolve_path(filepath))
        self.filepath = filepath
        self.data = self._load_json(filepath)

    def _load_json(self, path):
        """Safely load JSON data with validation."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Core knowledge file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {path}: {e}")

        if not isinstance(data, dict):
            raise ValueError("core_knowledge.json must contain a JSON object (key: answer).")

        return data

    def get(self, key: str) -> str | None:

        if not key:
            return None
        
        items = self.data.get(key, [])
        if not items:
            return None
        
        # Build final context text
        if key == "roles":
            title = "Different roles in the Toastmasters meeting include:"
        elif key == "pathways":
            title = "The available Toastmasters Pathways are:"
        else:
            title = f"{key.capitalize()} include:"

        bullet_list = "\n".join(f"• {item}" for item in items)
        return f"{title}\n{bullet_list}"

