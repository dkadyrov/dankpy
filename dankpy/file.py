import os
from datetime import datetime

class File:
    def init(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(self.filepath)
        self.extension = os.path.splitext(self.filepath)[1]
        self.directory = os.path.dirname(self.filepath)
        self.size = os.path.getsize(self.filepath)
        self.modified = datetime.fromtimestamp(os.path.getmtime(self.filepath))
        self.created = datetime.fromtimestamp(os.path.getctime(self.filepath))
        self.accessed = datetime.fromtimestamp(os.path.getatime(self.filepath))        

def metadata(filepath: str) -> dict:
    """
    Generates metadata of file

    Args:
        filepath (str): filepath of file

    Returns:
        dict: metadata of file
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
