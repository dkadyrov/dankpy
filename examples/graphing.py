#%%
from dankpy.graph import graph 

# Initialize the graph
fig = graph()

# Add a line to the graph
fig.add_scatter(x=[1,2,3,4], y=[1,2,3,4])

# Save the graph to a file
fig.save_latex('graphing.pdf')
# %%
