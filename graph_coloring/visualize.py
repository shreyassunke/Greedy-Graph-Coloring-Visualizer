"""Graph visualization with colored nodes using NetworkX, Matplotlib, and Plotly."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Union

from .greedy_color import greedy_color
from .adjacency import adj_matrix_to_list, adj_list_to_matrix, symmetrize_adj_list

PALETTE = [
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
    "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
    "#ccebc5", "#ffed6f",
]


def _get_adjacency_and_colors(
    A=None,
    E=None,
    n=None,
    directed=False,
    C_color=None,
    ignore_self_loops=False,
):
    """Resolve input format and compute colors if not provided."""
    if A is not None:
        A = np.asarray(A)
        n = A.shape[0]
        E = adj_matrix_to_list(A, directed=directed)
    elif E is not None:
        A = adj_list_to_matrix(E, n=n, directed=directed)
        n = A.shape[0]
    else:
        raise ValueError("Provide either A (adjacency matrix) or E (adjacency list)")

    if C_color is None:
        V_list = list(range(n))
        E_for_coloring = symmetrize_adj_list(E) if directed else E
        C_color = greedy_color(V_list, E_for_coloring, ignore_self_loops=ignore_self_loops)

    return A, E, n, C_color


def draw_colored_graph(
    A: Union[list[list[int]], np.ndarray] | None = None,
    E: list[list[int]] | None = None,
    n: int | None = None,
    *,
    directed: bool = False,
    C_color: list[int] | None = None,
    ignore_self_loops: bool = False,
    save_path: str | None = None,
    title: str | None = None,
    show: bool = True,
    one_indexed_labels: bool = False,
) -> None:
    """
    Draw a graph with nodes colored by the greedy coloring algorithm (Matplotlib).

    Args:
        A: Adjacency matrix (n×n). Provide A or E.
        E: Adjacency list. Provide A or E.
        n: Number of vertices (only used when E is provided and matrix size is ambiguous).
        directed: Whether the graph is directed.
        C_color: Pre-computed color assignment. If None, greedy_color is run.
        ignore_self_loops: Passed to greedy_color when C_color is None.
        save_path: Path to save the figure (e.g. 'graph.png' or 'graph.svg').
        title: Plot title.
        show: Whether to call plt.show().
        one_indexed_labels: If True, label vertices as V1, V2, ... (screenshot style).
    """
    A, _, n, C_color = _get_adjacency_and_colors(
        A=A, E=E, n=n, directed=directed, C_color=C_color, ignore_self_loops=ignore_self_loops
    )

    GraphClass = nx.DiGraph if directed else nx.Graph
    G = nx.from_numpy_array(A, create_using=GraphClass)

    pos = nx.spring_layout(G, seed=42)
    cmap = plt.cm.Set3
    num_colors = max(C_color) if C_color else 1
    node_colors = [cmap((c - 1) / max(num_colors, 1)) for c in C_color]

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=800,
        with_labels=True,
        labels={i: f"V{i + 1}" if one_indexed_labels else f"V{i}" for i in range(n)},
        font_size=12,
        font_weight="bold",
        edge_color="gray",
        width=2,
    )
    if title:
        ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def draw_colored_graph_interactive(
    A: Union[list[list[int]], np.ndarray] | None = None,
    E: list[list[int]] | None = None,
    n: int | None = None,
    *,
    directed: bool = False,
    C_color: list[int] | None = None,
    ignore_self_loops: bool = False,
    save_path: str | None = None,
    title: str | None = None,
) -> None:
    """
    Draw a graph with nodes colored by the greedy coloring algorithm (Plotly, interactive).

    Args:
        A: Adjacency matrix. Provide A or E.
        E: Adjacency list. Provide A or E.
        n: Number of vertices (only used when E is provided).
        directed: Whether the graph is directed.
        C_color: Pre-computed color assignment. If None, greedy_color is run.
        ignore_self_loops: Passed to greedy_color when C_color is None.
        save_path: Path to save HTML (e.g. 'graph.html').
        title: Plot title.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for interactive visualization. Install with: pip install plotly")

    A, _, n, C_color = _get_adjacency_and_colors(
        A=A, E=E, n=n, directed=directed, C_color=C_color, ignore_self_loops=ignore_self_loops
    )

    GraphClass = nx.DiGraph if directed else nx.Graph
    G = nx.from_numpy_array(A, create_using=GraphClass)
    pos = nx.spring_layout(G, seed=42)

    edge_trace_x, edge_trace_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace_x.extend([x0, x1, None])
        edge_trace_y.extend([y0, y1, None])

    node_x = [pos[i][0] for i in range(n)]
    node_y = [pos[i][1] for i in range(n)]

    num_colors = max(C_color) if C_color else 1
    colors = plt.cm.Set3([(c - 1) / max(num_colors, 1) for c in C_color])
    node_colors_rgba = [f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})" for r, g, b, a in colors]

    edge_trace = go.Scatter(
        x=edge_trace_x, y=edge_trace_y,
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[f"V{i}" for i in range(n)],
        textposition="top center",
        textfont=dict(size=12, color="black"),
        hovertext=[f"V{i}: color {C_color[i]}" for i in range(n)],
        hoverinfo="text",
        marker=dict(
            size=25,
            color=node_colors_rgba,
            line=dict(width=2, color="white"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        title=title or "Colored Graph",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=20, l=20, r=20, t=40),
        hovermode="closest",
    )
    if save_path:
        fig.write_html(save_path)
    fig.show()


def _color_for(c: int, num_colors: int) -> str:
    """Map a 1-based color index to a hex color from the palette."""
    return PALETTE[(c - 1) % len(PALETTE)]


def build_plotly_figure(
    A: Union[list[list[int]], np.ndarray],
    C_color: list[int],
    *,
    directed: bool = False,
    mode_3d: bool = True,
    dark_mode: bool = True,
    one_indexed_labels: bool = True,
    title: str | None = None,
) -> go.Figure:
    """
    Build an interactive Plotly figure (2D or 3D) for a colored graph.

    Args:
        A: Adjacency matrix (n x n).
        C_color: Pre-computed 1-based color list from greedy_color.
        directed: Whether the graph is directed.
        mode_3d: If True, render a 3D scene with Scatter3d (rotatable globe).
                 If False, render a 2D interactive Plotly chart.
        dark_mode: If True, use dark backgrounds; if False, light backgrounds.
        one_indexed_labels: Label vertices V1, V2, ... instead of V0, V1, ...
        title: Figure title.

    Returns:
        A plotly go.Figure ready for st.plotly_chart() or fig.show().
    """
    A = np.asarray(A)
    n = A.shape[0]
    num_colors = max(C_color) if C_color else 1

    GraphClass = nx.DiGraph if directed else nx.Graph
    G = nx.from_numpy_array(A, create_using=GraphClass)

    degrees = [G.degree(i) for i in range(n)]
    node_sizes = _degree_to_sizes(degrees, mode_3d=mode_3d)

    # Weight edges so the spring layout pushes apart large nodes.
    # Lower weight = weaker attraction = more separation at equilibrium.
    if G.number_of_edges() > 0:
        combined = {(u, v): node_sizes[u] + node_sizes[v] for u, v in G.edges()}
        min_c = min(combined.values())
        for (u, v), c in combined.items():
            G[u][v]["weight"] = (min_c / c) ** 2

    dim = 3 if mode_3d else 2
    k = 1.8 / (n ** 0.5)
    pos = nx.spring_layout(G, dim=dim, seed=42, k=k)

    labels = [f"V{i + 1}" if one_indexed_labels else f"V{i}" for i in range(n)]
    node_colors = [_color_for(C_color[i], num_colors) for i in range(n)]
    hover = [f"{labels[i]}: color {C_color[i]}, degree {degrees[i]}" for i in range(n)]

    theme = _DARK_THEME if dark_mode else _LIGHT_THEME

    if mode_3d:
        fig = _build_3d(G, pos, n, labels, node_colors, hover, title, theme, node_sizes)
    else:
        fig = _build_2d(G, pos, n, labels, node_colors, hover, title, theme, node_sizes)

    return fig


_DARK_THEME = dict(
    paper_bg="rgba(0,0,0,0)",
    plot_bg="#0e1117",
    scene_bg="#0e1117",
    text_color="white",
    edge_color="rgba(180,180,180,0.5)",
    title_color="white",
    template="plotly_dark",
)
_LIGHT_THEME = dict(
    paper_bg="white",
    plot_bg="white",
    scene_bg="#f0f2f6",
    text_color="#31333F",
    edge_color="rgba(100,100,100,0.5)",
    title_color="#31333F",
    template="plotly_white",
)


def _degree_to_sizes(degrees: list[int], *, mode_3d: bool) -> list[float]:
    """Map node degrees to proportional sizes.

    3D mode returns sphere radii; 2D mode returns marker pixel sizes.
    Nodes with zero edges get the minimum size.
    """
    if mode_3d:
        r_min, r_max = 0.04, 0.10
    else:
        s_min, s_max = 18.0, 45.0

    d_min = min(degrees)
    d_max = max(degrees)
    span = d_max - d_min if d_max > d_min else 1

    sizes = []
    for d in degrees:
        t = (d - d_min) / span
        if mode_3d:
            sizes.append(r_min + t * (r_max - r_min))
        else:
            sizes.append(s_min + t * (s_max - s_min))
    return sizes


def _sphere_mesh(cx, cy, cz, r=0.06, resolution=20):
    """Generate (x, y, z, i, j, k) arrays for a triangulated sphere mesh."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = cx + r * np.outer(np.cos(u), np.sin(v)).flatten()
    y = cy + r * np.outer(np.sin(u), np.sin(v)).flatten()
    z = cz + r * np.outer(np.ones_like(u), np.cos(v)).flatten()
    return x, y, z


def _build_3d(G, pos, n, labels, node_colors, hover, title, theme, node_sizes):
    node_x = [pos[i][0] for i in range(n)]
    node_y = [pos[i][1] for i in range(n)]
    node_z = [pos[i][2] for i in range(n)]

    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])
        edge_z.extend([pos[u][2], pos[v][2], None])

    traces = []

    traces.append(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(width=6, color=theme["edge_color"]),
        hoverinfo="none",
    ))

    for i in range(n):
        r = node_sizes[i]
        sx, sy, sz = _sphere_mesh(node_x[i], node_y[i], node_z[i], r=r)
        traces.append(go.Mesh3d(
            x=sx, y=sy, z=sz,
            alphahull=0,
            color=node_colors[i],
            opacity=1.0,
            lighting=dict(
                ambient=0.3,
                diffuse=0.8,
                specular=1.0,
                roughness=0.05,
                fresnel=0.4,
            ),
            lightposition=dict(x=100, y=200, z=300),
            hovertext=hover[i],
            hoverinfo="text",
            name=labels[i],
            showlegend=False,
        ))

    label_offsets = [r + 0.03 for r in node_sizes]
    traces.append(go.Scatter3d(
        x=node_x,
        y=node_y,
        z=[node_z[i] + label_offsets[i] for i in range(n)],
        mode="text",
        text=labels,
        textfont=dict(size=14, color=theme["text_color"]),
        hoverinfo="none",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template=theme["template"],
        showlegend=False,
        title=dict(text=title or "Greedy Coloring Algorithm", font=dict(color=theme["title_color"])),
        paper_bgcolor=theme["paper_bg"],
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor=theme["scene_bg"],
        ),
    )
    return fig


def _build_2d(G, pos, n, labels, node_colors, hover, title, theme, node_sizes):
    node_x = [pos[i][0] for i in range(n)]
    node_y = [pos[i][1] for i in range(n)]

    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=2, color=theme["edge_color"]),
        hoverinfo="none",
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.95,
            line=dict(width=2, color="white"),
        ),
        text=labels,
        textposition="top center",
        textfont=dict(size=12, color=theme["text_color"]),
        hovertext=hover,
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        template=theme["template"],
        showlegend=False,
        title=dict(text=title or "Greedy Coloring Algorithm", font=dict(color=theme["title_color"])),
        paper_bgcolor=theme["paper_bg"],
        plot_bgcolor=theme["plot_bg"],
        margin=dict(l=0, r=0, b=0, t=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest",
    )
    return fig
