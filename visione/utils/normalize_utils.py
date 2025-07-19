import re
import unicodedata
import os

def normalize_filename(name: str) -> str:
    """
    Normalize a filename by:
      - Replacing spaces and special characters with underscores
      - Removing or replacing Umlaute and accented characters
      - Removing non-ASCII characters
    """
    # Replace Umlaute and accented chars
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    # Replace spaces and special chars with underscore
    name = re.sub(r'[^\w.-]', '_', name)
    # Remove multiple underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores or dots
    name = name.strip('._')
    return name

def normalize_path(path: str) -> str:
    """
    Normalize all parts of a path (folders and file) recursively.
    """
    parts = []
    for part in os.path.normpath(path).split(os.sep):
        parts.append(normalize_filename(part))
    return os.sep.join(parts)
