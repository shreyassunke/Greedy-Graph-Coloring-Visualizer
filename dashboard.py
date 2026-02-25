"""Interactive Streamlit dashboard for greedy graph coloring."""

import streamlit as st
import numpy as np

from graph_coloring.greedy_color import greedy_color
from graph_coloring.adjacency import adj_matrix_to_list, has_self_loops_in_matrix, symmetrize_adj_list
from graph_coloring.visualize import build_plotly_figure, PALETTE

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Graph Coloring", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    n = st.slider("Number of vertices", min_value=2, max_value=15, value=5)
    directed = st.checkbox("Directed graph")
    ignore_self_loops = st.checkbox("Ignore self-loops (skip instead of error)")
    edge_probability = st.slider(
        "Edge probability (for randomize)",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
    )
    st.divider()
    mode_3d = st.toggle("3D Mode", value=True)
    dark_mode = st.toggle("Dark Mode", value=False)

# ---------------------------------------------------------------------------
# Dark-mode CSS override (light is the native default from config.toml)
# When dark_mode=True, inject dark theme so data editor & checkboxes match.
# ---------------------------------------------------------------------------
if dark_mode:
    st.markdown("""<style>
    /* ---- Streamlit CSS custom properties ---- */
    :root {
        --primary-color: #ff4b4b;
        --background-color: #0e1117;
        --secondary-background-color: #262730;
        --text-color: #fafafa;
        --font: "Source Sans Pro", sans-serif;
        color-scheme: dark;
    }

    /* ---- Global surfaces ---- */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stBottom"] {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* ---- All text elements ---- */
    h1, h2, h3, h4, h5, h6,
    p, span, label, li, td, th, summary, small,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stCaption"],
    [data-testid="stText"],
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] div {
        color: #fafafa !important;
    }

    /* ---- Checkbox: visible box with light border on dark ---- */
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] label span,
    [data-testid="stCheckbox"] label p {
        color: #fafafa !important;
    }
    [data-testid="stCheckbox"] [data-baseweb="checkbox"] {
        background-color: transparent !important;
    }
    [data-testid="stCheckbox"] [role="checkbox"] {
        border-color: rgba(250, 250, 250, 0.6) !important;
        background-color: transparent !important;
    }
    [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }

    /* ---- Toggle switch: visible track ---- */
    [data-testid="stToggle"] label,
    [data-testid="stToggle"] label span,
    [data-testid="stToggle"] label p {
        color: #fafafa !important;
    }
    [data-testid="stToggle"] [role="checkbox"] {
        background-color: rgba(250, 250, 250, 0.3) !important;
    }
    [data-testid="stToggle"] [role="checkbox"][aria-checked="true"] {
        background-color: #ff4b4b !important;
    }
    [data-testid="stToggle"] [role="checkbox"] div {
        background-color: #ffffff !important;
    }

    /* ---- Slider ---- */
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] div[data-baseweb] div {
        color: #fafafa !important;
    }

    /* ---- Buttons ---- */
    [data-testid="stBaseButton-secondary"] {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #555 !important;
    }
    [data-testid="stBaseButton-secondary"]:hover {
        background-color: #333 !important;
        border-color: #666 !important;
    }

    /* ---- Expander ---- */
    [data-testid="stExpander"] {
        background-color: #262730 !important;
        border-color: #555 !important;
    }

    /* ---- Data editor / glide-data-grid: dark theme variables ---- */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    div[data-testid="stDataFrame"] {
        --gdg-bg-cell: #1e1e1e;
        --gdg-bg-cell-medium: #262730;
        --gdg-bg-header: #262730;
        --gdg-bg-header-has-focus: #333;
        --gdg-bg-header-hovered: #2d2d2d;
        --gdg-bg-bubble: #333;
        --gdg-bg-bubble-selected: #444;
        --gdg-bg-search-result: #3d3d00;
        --gdg-border-color: rgba(255, 255, 255, 0.12);
        --gdg-drilldown-border: rgba(255, 255, 255, 0.35);
        --gdg-link-color: #6eb3f7;
        --gdg-text-dark: #fafafa;
        --gdg-text-medium: #d0d0d0;
        --gdg-text-light: #9ca3af;
        --gdg-text-bubble-selected: #fafafa;
        --gdg-text-header: #fafafa;
        --gdg-text-header-selected: #ffffff;
        background-color: #1e1e1e !important;
    }
    [data-testid="stDataFrame"] * {
        color: #fafafa !important;
    }

    /* ---- Alert boxes ---- */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: inherit !important;
    }
    </style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("Greedy Graph Coloring")
st.caption("Enter an adjacency matrix or generate a random one, then hit **Randomize** to see the coloring.")

# ---------------------------------------------------------------------------
# Session state: matrix
# ---------------------------------------------------------------------------
if "matrix" not in st.session_state or st.session_state.get("prev_n") != n:
    st.session_state.matrix = np.zeros((n, n), dtype=int)
    st.session_state.prev_n = n


def _randomize():
    rng = np.random.default_rng()
    A = (rng.random((n, n)) < edge_probability).astype(int)
    np.fill_diagonal(A, 0)
    if not directed:
        A = np.triu(A, 1)
        A = A + A.T
    st.session_state.matrix = A


# ---------------------------------------------------------------------------
# Matrix input area
# ---------------------------------------------------------------------------
col_input, col_graph = st.columns([1, 1.4], gap="large")

with col_input:
    st.subheader("Adjacency Matrix")

    btn_cols = st.columns(2)
    with btn_cols[0]:
        st.button("Randomize Matrix", on_click=_randomize, width="stretch")
    with btn_cols[1]:
        clear = st.button("Clear Matrix", width="stretch")
        if clear:
            st.session_state.matrix = np.zeros((n, n), dtype=int)

    st.markdown(
        "<small>Click cells to toggle edges (0 ↔ 1). Rows = source, columns = target.</small>",
        unsafe_allow_html=True,
    )

    edited = st.data_editor(
        st.session_state.matrix.tolist(),
        width="stretch",
        num_rows="fixed",
        column_config={
            str(j): st.column_config.NumberColumn(
                label=f"V{j+1}",
                min_value=0, max_value=1, step=1, default=0,
            )
            for j in range(n)
        },
        key="matrix_editor",
    )

    A = np.array(edited, dtype=int)
    st.session_state.matrix = A

# ---------------------------------------------------------------------------
# Graph output area (always renders -- auto-updates on any setting change)
# ---------------------------------------------------------------------------
with col_graph:
    st.subheader("Colored Graph")

    has_loops = has_self_loops_in_matrix(A)

    if has_loops and not ignore_self_loops:
        st.error(
            "The matrix has self-loops (non-zero diagonal). "
            "A graph with self-loops cannot be properly colored. "
            "Enable **Ignore self-loops** in the sidebar or zero-out the diagonal."
        )
    else:
        E = adj_matrix_to_list(A, directed=directed)
        E_for_coloring = symmetrize_adj_list(E) if directed else E
        V_list = list(range(n))

        try:
            C_color = greedy_color(V_list, E_for_coloring, ignore_self_loops=ignore_self_loops)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        fig = build_plotly_figure(
            A,
            C_color,
            directed=directed,
            mode_3d=mode_3d,
            dark_mode=dark_mode,
            one_indexed_labels=True,
        )
        st.plotly_chart(fig, width="stretch")

        # ---- Summary metrics ----
        num_colors = max(C_color)
        num_edges = int(np.sum(A) // (1 if directed else 2))

        m1, m2, m3 = st.columns(3)
        m1.metric("Vertices", n)
        m2.metric("Edges", num_edges)
        m3.metric("Colors used", num_colors)

        # ---- Color table ----
        st.markdown("**Color assignment**")
        rows = []
        for i in range(n):
            c = C_color[i]
            hex_color = PALETTE[(c - 1) % len(PALETTE)]
            rows.append(
                f"| V{i+1} | "
                f'<span style="display:inline-block;width:16px;height:16px;'
                f"border-radius:50%;background:{hex_color};"
                f'vertical-align:middle;margin-right:4px"></span> '
                f"Color {c} |"
            )

        table_md = "| Vertex | Color |\n|--------|-------|\n" + "\n".join(rows)
        st.markdown(table_md, unsafe_allow_html=True)

        # ---- Adjacency list ----
        with st.expander("Adjacency list"):
            for i, neighbors in enumerate(E):
                st.text(f"  V{i+1} → {[j+1 for j in neighbors]}")
