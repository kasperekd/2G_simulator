"""
Module for loading configuration files (JSON or YAML).

Provides the `ConfigLoader` class with static methods for:
- loading configurations from JSON/YAML files without validation;
- retrieving the raw content of a configuration file as a string (for debugging).

Classes:
    ConfigLoader: Class with static methods for loading configuration data.
"""

import json
import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Configuration file loader (JSON/YAML) without validation."""

    @staticmethod
    def load(file_path: str | Path) -> Dict[str, Any]:
        """Loads a configuration file without validating its contents.

        Args:
            file_path (str | Path): Path to the configuration file (.json/.yaml/.yml).

        Returns:
            Dict[str, Any]: Dictionary containing raw configuration data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
            json.JSONDecodeError / yaml.YAMLError: On parsing errors.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ('.yaml', '.yml'):
                return yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

    @staticmethod
    def load_raw(file_path: str | Path) -> str:
        """Reads a configuration file as plain text (for debugging).

        Args:
            file_path (str | Path): Path to the file.

        Returns:
            str: Raw file contents as a string.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
