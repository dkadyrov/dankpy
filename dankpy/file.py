import os 
import pathlib
from datetime import datetime 

def metadata(filepath):
    """
    Returns metadata for a file.
    """
    metadata = {}
    metadata["filepath"] = filepath
    metadata["filename"] = os.path.basename(filepath)
    metadata["extension"] = os.path.splitext(filepath)[1]
    metadata["directory"] = os.path.dirname(filepath)
    metadata["size"] = os.path.getsize(filepath)
    metadata["modified"] = datetime.fromtimestamp(os.path.getmtime(filepath))
    metadata["created"] = datetime.fromtimestamp(os.path.getctime(filepath))
    metadata["accessed"] = datetime.fromtimestamp(os.path.getatime(filepath))

    return metadata