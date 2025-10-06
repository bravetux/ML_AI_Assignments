"""
Draw a neural network diagram for the Pima Indians Diabetes prediction problem.

The Pima dataset has 8 input features:
  - Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
    BMI, DiabetesPedigreeFunction, Age

This script draws a simple feed-forward architecture with a configurable
hidden-layer layout and saves the diagram as `nn_diagram.png` in the same
folder.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple
from pathlib import Path


def draw_neural_net(ax, left: float, right: float, bottom: float, top: float, layer_sizes: List[int], feature_labels: List[str] | None = None):
    """Draw a neural network cartoon using matplolib.

    Parameters
    - ax: matplotlib Axes
    - left, right, bottom, top: bounds of drawing area
    - layer_sizes: list of layer sizes, e.g. [8, 12, 8, 1]
    """
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes) + 1)
    h_spacing = (right - left) / float(n_layers - 1)

    # positions of nodes: list of lists
    node_coords: List[List[Tuple[float, float]]] = []
    for i, layer_size in enumerate(layer_sizes):
        layer_x = left + i * h_spacing
        # center layer vertically
        layer_top = bottom + (max(layer_sizes) - layer_size) * v_spacing / 2.0
        coords = []
        for j in range(layer_size):
            y = top - (j + 1) * v_spacing - (max(layer_sizes) - layer_size) * v_spacing / 2.0
            coords.append((layer_x, y))
        node_coords.append(coords)

    # pick colors for layers: input, hidden(s), output
    cmap = plt.get_cmap('tab10')
    layer_colors = []
    for i in range(n_layers):
        if i == 0:
            layer_colors.append('#b7e4c7')  # light green for inputs
        elif i == n_layers - 1:
            layer_colors.append('#ffadad')  # light red for output
        else:
            # cycle through categorical colors for hidden layers
            layer_colors.append(cmap((i - 1) % 10))

    # draw connections (light gray) — draw first so nodes overlay them
    for i in range(n_layers - 1):
        for (x1, y1) in node_coords[i]:
            for (x2, y2) in node_coords[i + 1]:
                ax.plot([x1, x2], [y1, y2], color='#8d99ae', linewidth=0.7, alpha=0.6, zorder=1)

    # draw nodes with colors per layer
    for i, layer in enumerate(node_coords):
        col = layer_colors[i]
        for (x, y) in layer:
            circle = Circle((x, y), v_spacing * 0.18, fill=True, facecolor=col, ec='#2b2d42', linewidth=0.8, zorder=4)
            ax.add_patch(circle)

    # add input feature labels if provided
    if feature_labels is not None:
        if len(feature_labels) >= len(node_coords[0]):
            for (x, y), label in zip(node_coords[0], feature_labels):
                ax.text(x - 0.03, y, label, fontsize=8, ha='right', va='center')

    # add labels for layers
    ax.text(left, top + v_spacing * 0.6, 'Input layer (8 features)', fontsize=10, ha='left')
    ax.text(right, top + v_spacing * 0.6, 'Output (Diabetes: yes/no)', fontsize=10, ha='right')


def main() -> None:
    # Pima features: 8 inputs
    input_size = 8
    # Example architecture: two hidden layers
    layer_sizes = [input_size, 12, 8, 1]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    feature_labels = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigree', 'Age'
    ]
    draw_neural_net(ax, .1, .9, .1, .9, layer_sizes, feature_labels=feature_labels)

    ax.set_title('Neural Network for Pima Indians Diabetes Prediction')

    # save the figure next to this python file
    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / 'nn_diagram.svg'
    # save as SVG (vector) for scalable quality
    fig.savefig(out_path, bbox_inches='tight', format='svg')
    print(f"Saved neural network diagram to: {out_path}")


if __name__ == '__main__':
    main()
