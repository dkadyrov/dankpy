import os
import re
from dankpy import dt, dankframe
import librosa 
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

        if self.extension in audiofiles:
            try: 
                self.channel = int(re.findall("\d+", self.filename.split("_")[-2])[0])
            except:
                pass

            try: 
                self.record_number = int(self.filename.split("_")[-1].split(".")[0])
            except: 
                pass 

            self.sample_rate = librosa.get_samplerate(self.filepath)
            self.duration = librosa.get_duration(path=self.filepath)

            # TODO - Parse datetime from filename
            self.start = dt.read_datetime(self.filename[:23])
            self.end = self.start + dt.timedelta(seconds=self.duration)

    def metadata(self) -> dict: 
        metadata = {
            "filepath": self.filepath,
            "filename": self.filename,
            "extension": self.extension,
            "directory": self.directory,
            "size": self.size,
            "modified": self.modified,
            "created": self.created,
            "accessed": self.accessed,
        }    

        if self.extension in audiofiles:
            metadata["channel"] = self.channel
            metadata["record_number"] = self.record_number
            metadata["sample_rate"] = self.sample_rate
            metadata["duration"] = self.duration
            metadata["start"] = self.start
            metadata["end"] = self.end
        
        return metadata
            
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

    def detect_delimiter(self, n: int = 2) -> str:
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


audiofiles = [
    "wav",
    "mp3",
    "aiff",
    "flac",
    "ogg",
    "wma",
    "m4a",
    "aac",
    "alac",
    "aif",
    "aifc",
    "aiffc",
    "au",
    "snd",
    "cdda",
    "raw",
    "mpc",
    "vqf",
    "tta",
    "wv",
    "ape",
    "ac3",
    "dts",
    "dtsma",
    "dtshr",
    "dtshd",
    "eac3",
    "thd",
    "thd+ac3",
    "thd+dts",
    "thd+dd",
    "thd+dd+ac3",
    "thd+dd+dts",
    "thd+dd+dtsma",
    "thd+dd+dtshr",
    "thd+dd+dtshd",
]


def metadata(filepath: str, extended=False) -> dict:
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
    metadata["extension"] = os.path.splitext(filepath)[1].replace(".", "")
    metadata["directory"] = os.path.dirname(filepath)
    metadata["size"] = os.path.getsize(filepath)
    metadata["modified"] = datetime.fromtimestamp(os.path.getmtime(filepath))
    metadata["created"] = datetime.fromtimestamp(os.path.getctime(filepath))
    metadata["accessed"] = datetime.fromtimestamp(os.path.getatime(filepath))

    if extended:
        if metadata["extension"] in audiofiles:
            try: 
                metadata["channel"] = int(
                    re.findall("\d+", metadata["filename"].split("_")[-2])[0]
                )
            except:
                metadata["channel"] = None

            try: 
                metadata["sample_rate"] = librosa.get_samplerate(metadata["filepath"])
            except:
                metadata["sample_rate"] = None

            try:  
                metadata["duration"] = librosa.get_duration(path=metadata["filepath"])
            except:
                metadata["duration"] = None

            try: 
                metadata["record_number"] = int(metadata["filename"].split("_")[-1].split(".")[0])
            except:
                metadata["record_number"] = None
                
            try: 
                metadata["start"] = dt.read_datetime(metadata["filename"][:23])
            except:
                metadata["start"]  = None

            try: 
                metadata["end"] = metadata["start"] + dt.timedelta(seconds=metadata["duration"])
            except: 
                metadata["end"] = None

    return metadata

def metadatas(filepaths: list, extended=False) -> dankframe.DankFrame:
    """
    Generates metadata of files

    Args:
        filepaths (list): list of filepaths

    Returns:
        pd.DataFrame: metadata of files
    """

    return dankframe.pd.DataFrame([metadata(filepath, extended) for filepath in filepaths])