"""Demo script for greedy graph coloring: algorithm, adjacency conversion, and visualization."""

import numpy as np
from graph_coloring import greedy_color, adj_matrix_to_list, adj_list_to_matrix
from graph_coloring.visualize import draw_colored_graph, draw_colored_graph_interactive


def main():
    # Example: 4-vertex graph from screenshot (V1-V4, triangle V1-V2-V3 + V2-V4)
    # Edges: V1-V2, V1-V3, V2-V3, V2-V4
    A = np.array([
        [0, 1, 1, 0],  # V1: adjacent to V2, V3
        [1, 0, 1, 1],  # V2: adjacent to V1, V3, V4
        [1, 1, 0, 0],  # V3: adjacent to V1, V2
        [0, 1, 0, 0],  # V4: adjacent to V2
    ])

    print("=== Adjacency Matrix ===")
    print(A)
    print()

    # Step 2: Convert to adjacency list
    E = adj_matrix_to_list(A, directed=False)
    print("=== Adjacency List ===")
    for i, neighbors in enumerate(E):
        print(f"  V{i + 1}: {[j + 1 for j in neighbors]}")
    print()

    # Step 1: Greedy coloring
    n = A.shape[0]
    V_list = list(range(n))
    C_color = greedy_color(V_list, E)
    print("=== Color Assignment (matches screenshot) ===")
    for i in range(n):
        print(f"  V{i + 1}: color {C_color[i]} (index {C_color[i] - 1})")
    print(f"  Colors used: {max(C_color)}")
    print()

    # Step 3: Visualize (Matplotlib)
    print("Saving Matplotlib visualization to graph.png...")
    draw_colored_graph(
        A, directed=False,
        save_path="graph.png",
        title="Greedy Coloring Algorithm",
        show=False,
        one_indexed_labels=True,
    )

    # Optional: Interactive Plotly visualization
    # print("Opening Plotly interactive visualization...")
    # draw_colored_graph_interactive(A, directed=False, save_path="graph.html")


def demo_adjacency_list_input():
    """Demo using adjacency list as input."""
    E = [[1, 2], [0, 2], [0, 1]]  # Triangle
    A = adj_list_to_matrix(E, directed=False)
    print("=== From Adjacency List ===")
    print("E =", E)
    print("A =")
    print(A)
    C_color = greedy_color(list(range(3)), E)
    print("Colors:", C_color)
    draw_colored_graph(E=E, save_path="graph_from_list.png", title="Triangle Graph")


if __name__ == "__main__":
    main()
    # demo_adjacency_list_input()
