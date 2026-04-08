"""
scripts/graph_visualizer.py
============================
Automatic graph visualization for emotion trajectories.

Generates a directed graph from a list of OCC emotion labels and saves it
as a PNG file.  Uses networkx for graph construction and matplotlib for
rendering.  No display is shown — output is file-only.
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import networkx as nx


def visualize_trajectory(trajectory: list, save_path: str) -> None:
    """
    Generate a directed graph visualization for *trajectory* and save as PNG.

    Each unique emotion in the trajectory becomes a node.  Directed edges are
    drawn for each consecutive transition pair.  If the same transition appears
    more than once (e.g. a loop), the edge is drawn once with a weight label.

    The output directory is created automatically if it does not exist.

    Args:
        trajectory (list[str]): Ordered list of OCC emotion labels.
                                Must contain at least 2 elements.
        save_path  (str):       Absolute or relative path for the PNG output
                                (e.g. "outputs/sample_1_graph.png").

    Raises:
        ValueError: If trajectory contains fewer than 2 elements.
    """
    if len(trajectory) < 2:
        raise ValueError(
            "trajectory must contain at least 2 emotions to visualize transitions."
        )

    # Ensure output directory exists
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── Build directed graph ──────────────────────────────────────────────────
    G = nx.DiGraph()

    for emotion in trajectory:
        G.add_node(emotion)

    edge_counts: dict = {}
    for i in range(len(trajectory) - 1):
        src = trajectory[i]
        tgt = trajectory[i + 1]
        key = (src, tgt)
        edge_counts[key] = edge_counts.get(key, 0) + 1

    for (src, tgt), count in edge_counts.items():
        G.add_edge(src, tgt, weight=count)

    # ── Layout ────────────────────────────────────────────────────────────────
    pos = nx.spring_layout(G, seed=42)  # fixed seed for reproducibility

    # ── Color nodes by trajectory position ───────────────────────────────────
    unique_nodes  = list(G.nodes())
    first_emotion = trajectory[0]
    last_emotion  = trajectory[-1]

    node_colors = []
    for node in unique_nodes:
        if node == first_emotion:
            node_colors.append("#4C9BE8")   # blue  — start
        elif node == last_emotion:
            node_colors.append("#E86B4C")   # red   — end
        else:
            node_colors.append("#7DC67D")   # green — intermediate

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(
        "Emotion Trajectory Graph\n"
        f"{' → '.join(trajectory)}",
        fontsize=13,
        pad=16,
    )

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_size=3000,
        node_color=node_colors,
        font_size=10,
        font_weight="bold",
        font_color="white",
        arrows=True,
        arrowsize=25,
        edge_color="#555555",
        width=2.0,
        connectionstyle="arc3,rad=0.1",
    )

    # Edge weight labels (shown only when an edge repeats)
    edge_labels = {
        (s, t): f"×{w}" for (s, t), w in
        nx.get_edge_attributes(G, "weight").items()
        if w > 1
    }
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax, font_size=9
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C9BE8", label="Start"),
        Patch(facecolor="#7DC67D", label="Intermediate"),
        Patch(facecolor="#E86B4C", label="End"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)