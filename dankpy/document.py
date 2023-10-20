#%%
from docx import Document
from docx.shared import Inches
import matplotlib.pyplot as plt
from io import BytesIO

# class Document():
#     def __init__(self):
#         self = Doc()

def add_dataframe(doc, df):
    t = doc.add_table(df.shape[0]+1, df.shape[1])
    for j in range(df.shape[-1]):
        t.cell(0,j).text = df.columns[j]
    for i in range(df.shape[0]):
        for j in range(df.shape[-1]):
            t.cell(i+1,j).text = str(df.values[i,j])

    return t         

def add_figure(doc, fig, ax, caption, add_break=True):
    memfile = BytesIO()
    fig.tight_layout()
    fig.savefig(memfile)
    doc.add_picture(memfile, width=Inches(6))
    plt.close(fig)
    
    if add_break:
        doc.add_page_break()

# document = Document()
#%%