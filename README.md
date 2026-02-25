# Greedy Graph Coloring

A Python implementation of the greedy graph coloring algorithm with adjacency matrix/list conversion and visualization.

## Features

- **Greedy coloring**: Assigns the smallest available color (1-based) to each vertex so no adjacent vertices share a color
- **Self-loop handling**: Detects self-loops and raises an error by default (uncolorable); optional flag to ignore
- **Adjacency conversion**: Convert between adjacency matrix and adjacency list
- **Directed/undirected**: Support for both graph types via a parameter
- **Visualization**: Matplotlib (static) and Plotly (interactive) backends

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### From adjacency matrix

```python
import numpy as np
from graph_coloring import greedy_color, adj_matrix_to_list, draw_colored_graph

A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
E = adj_matrix_to_list(A, directed=False)
C_color = greedy_color(list(range(3)), E)
print(C_color)  # e.g. [1, 2, 3]
draw_colored_graph(A, save_path="graph.png")
```

### From adjacency list

```python
from graph_coloring import greedy_color, adj_list_to_matrix, draw_colored_graph

E = [[1, 2], [0, 2], [0, 1]]
A = adj_list_to_matrix(E)
C_color = greedy_color(list(range(3)), E)
draw_colored_graph(E=E, save_path="graph.png")
```

### Interactive visualization (Plotly)

```python
from graph_coloring import draw_colored_graph_interactive

draw_colored_graph_interactive(A, save_path="graph.html")
```

### Self-loops

By default, graphs with self-loops raise `ValueError`:

```python
E = [[1, 0], [0]]  # self-loop at vertex 0
greedy_color([0, 1], E)  # ValueError: Graph contains self-loop; uncolorable
```

To ignore self-loops:

```python
greedy_color([0, 1], E, ignore_self_loops=True)
```

## Project structure

```
graph_coloring/
├── greedy_color.py   # Algorithm + self-loop handling
├── adjacency.py      # Matrix ↔ list conversion
└── visualize.py      # NetworkX + Matplotlib/Plotly
main.py               # Demo script
```

## Run demo

```bash
python main.py
```

This prints the adjacency matrix, adjacency list, color assignment, and saves a visualization to `graph.png`.
