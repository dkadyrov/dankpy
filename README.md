# dankpy

## About

My name is Daniel Kadyrov and I created the Dan-K-py repo and package for functions I created and use across projects. 

## Installation

Place your terminal/command prompt into the downloaded folder and enter the following command:

```bash
pip install . 
```

## Example of Graph Function

```python
from dankpy.graph import graph 

# Initialize the graph
fig = graph()

# Add a line to the graph
fig.add_scatter(x=[1,2,3,4], y=[1,2,3,4])

# Save the graph to a file
fig.save_latex('graphing.pdf')
```