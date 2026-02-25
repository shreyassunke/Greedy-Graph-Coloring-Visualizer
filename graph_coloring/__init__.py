"""Graph coloring module: greedy algorithm, adjacency conversion, and visualization."""

from .greedy_color import greedy_color
from .adjacency import adj_matrix_to_list, adj_list_to_matrix

__all__ = [
    "greedy_color",
    "adj_matrix_to_list",
    "adj_list_to_matrix",
    "draw_colored_graph",
    "draw_colored_graph_interactive",
    "build_plotly_figure",
]


def __getattr__(name):
    """Lazy import for visualization (requires matplotlib/plotly)."""
    if name == "draw_colored_graph":
        from .visualize import draw_colored_graph
        return draw_colored_graph
    if name == "draw_colored_graph_interactive":
        from .visualize import draw_colored_graph_interactive
        return draw_colored_graph_interactive
    if name == "build_plotly_figure":
        from .visualize import build_plotly_figure
        return build_plotly_figure
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
