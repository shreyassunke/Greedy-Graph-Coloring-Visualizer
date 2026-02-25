"""Conversion between adjacency matrix and adjacency list representations."""

import numpy as np
from typing import Union


def adj_matrix_to_list(
    A: Union[list[list[int]], np.ndarray],
    *,
    directed: bool = False,
) -> list[list[int]]:
    """
    Convert an adjacency matrix to an adjacency list.

    Args:
        A: n×n adjacency matrix. A[i][j] != 0 means an edge from i to j.
           A[i][i] = 1 indicates a self-loop at vertex i.
        directed: If False, treat the graph as undirected (edge i-j implies j-i).
                  If True, preserve matrix as-is.

    Returns:
        E: Adjacency list where E[i] = [j for all j where A[i][j] != 0].
           For undirected graphs, E[i] includes j if A[i][j] != 0 or A[j][i] != 0.
    """
    A = np.asarray(A)
    n = A.shape[0]
    E = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                if j not in E[i]:
                    E[i].append(j)
            if not directed and i != j and A[j, i] != 0:
                if j not in E[i]:
                    E[i].append(j)

    return E


def adj_list_to_matrix(
    E: list[list[int]],
    n: int | None = None,
    *,
    directed: bool = False,
) -> np.ndarray:
    """
    Convert an adjacency list to an adjacency matrix.

    Args:
        E: Adjacency list where E[i] = list of neighbors of vertex i.
        n: Number of vertices. If None, inferred from len(E) and max vertex index.
        directed: If False, ensure symmetry (edge i-j implies A[i][j]=A[j][i]=1).
                  If True, A[i][j]=1 only when j in E[i].

    Returns:
        A: n×n adjacency matrix. A[i][j] = 1 if there is an edge from i to j.
    """
    if n is None:
        n = max(len(E), max((v for neighbors in E for v in neighbors), default=-1) + 1)
        n = max(n, len(E))

    A = np.zeros((n, n), dtype=int)

    for i in range(len(E)):
        for j in E[i]:
            if 0 <= j < n:
                A[i, j] = 1
                if not directed and i != j:
                    A[j, i] = 1

    return A


def symmetrize_adj_list(E: list[list[int]]) -> list[list[int]]:
    """
    Return a symmetric copy of an adjacency list.

    For every edge i->j, ensures j->i also exists. This is needed because
    graph coloring requires that ANY edge between two vertices (regardless of
    direction) forces different colors.
    """
    n = len(E)
    sym = [list(neighbors) for neighbors in E]
    for i in range(n):
        for j in E[i]:
            if j < n and i not in sym[j]:
                sym[j].append(i)
    return sym


def has_self_loops_in_matrix(A: Union[list[list[int]], np.ndarray]) -> bool:
    """Check if the adjacency matrix has any self-loops (non-zero diagonal)."""
    A = np.asarray(A)
    return bool(np.any(np.diag(A) != 0))
