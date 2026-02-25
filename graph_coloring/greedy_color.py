"""Greedy graph coloring algorithm with self-loop handling."""


def _has_self_loops(E: list[list[int]], n: int) -> bool:
    """Check if the graph has any self-loops (vertex i adjacent to itself)."""
    for i in range(n):
        if i < len(E) and i in E[i]:
            return True
    return False


def greedy_color(
    V_list: list[int],
    E: list[list[int]],
    *,
    ignore_self_loops: bool = False,
) -> list[int]:
    """
    Greedy graph coloring algorithm.

    Assigns the smallest available color (1-based) to each vertex such that
    no two adjacent vertices share the same color.

    Args:
        V_list: List of vertex indices (e.g. [0, 1, ..., n-1]).
        E: Adjacency list where E[i] is the list of neighbors of vertex i.
        ignore_self_loops: If False (default), raise ValueError when the graph
            contains self-loops (uncolorable). If True, ignore self-loops.

    Returns:
        C_color: List where C_color[i] is the color (1-based) assigned to vertex i.

    Raises:
        ValueError: If the graph contains self-loops and ignore_self_loops is False.
    """
    n = len(V_list)
    C_color = [0] * n
    C_color[0] = 1

    if not ignore_self_loops and _has_self_loops(E, n):
        raise ValueError("Graph contains self-loop; uncolorable")

    for i in V_list:
        neighbor_colors = set()
        for j in E[i] if i < len(E) else []:
            if j == i and ignore_self_loops:
                continue
            if C_color[j] != 0:
                neighbor_colors.add(C_color[j])

        k = 1
        while k in neighbor_colors:
            k += 1
        C_color[i] = k

    return C_color
