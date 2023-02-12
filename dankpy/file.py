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

    def head(self, n: int) -> list:
        """
        Returns first n lines of file

        Args:
            n (int): number of lines to return

        Returns:
            list: first n lines of file
        """
        try:
            with open(self.filename) as f:
                head_lines = [next(f).rstrip() for x in range(n)]
        except StopIteration:
            with open(self.filename) as f:
                head_lines = f.read().splitlines()
        return head_lines

    def detect_delimiter(self, n:int=2) -> str:
        """
        Detects delimiter of file

        Args:
            n (int, optional): number of lines to check. Defaults to 2.

        Returns:
            str: delimiter of file
        """

        sample_lines = self.head(n)
        common_delimiters = [",", ";", "\t", " ", "|", ":"]
        for d in common_delimiters:
            ref = sample_lines[0].count(d)
            if ref > 0:
                if all([ref == sample_lines[i].count(d) for i in range(1, n)]):
                    return d
        return ","


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
