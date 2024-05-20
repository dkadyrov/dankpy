
from dankpy import file
import glob 
from pathlib import Path 

files = glob.glob(r"C:/Users/daniel/Desktop/conference/**/*.wav", recursive=True)

f = file.metadata(files[0], extended=True)
#%%
f
# files = file.metadatas(files, extended=True, stevens=True)

