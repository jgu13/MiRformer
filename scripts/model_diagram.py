import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJ_HOME = os.path.expanduser("~/projects/mirLM")

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

# Define colors and positions
colors = ['#B3CDE0', '#6497B1', '#005B96', '#03396C', '#011F4B']

# Rectangles (components)
components = [
    ("Padded mRNA Seq", 0.1, 0.1, 0.3, 0.08),
    ("Neural Network", 0.42, 0.1, 0.16, 0.08),
    ("Padded miRNA Seq", 0.6, 0.1, 0.3, 0.08),
    ("Feature Extractor\n(HyenaDNA/Transformer)", 0.3, 0.25, 0.4, 0.12),
    ("Concat Seq Embedding", 0.2, 0.42, 0.6, 0.08),
    ("miRNA-length Embedding", 0.4, 0.55, 0.4, 0.08),
    ("Average Pooling", 0.45, 0.68, 0.3, 0.06),
    ("3-layer MLP", 0.45, 0.78, 0.3, 0.1)
]

# Draw rectangles
for i, (label, x, y, w, h) in enumerate(components):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", ec='black', fc=colors[i % len(colors)], alpha=0.7)
    ax.add_patch(rect)
    plt.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10, color='black')

# Arrows to show flow
arrows = [
    (0.5, 0.18, 0.5, 0.25),
    (0.5, 0.37, 0.5, 0.42),
    (0.5, 0.50, 0.5, 0.55),
    (0.5, 0.63, 0.5, 0.68),
    (0.5, 0.74, 0.5, 0.78)
]

for (x1, y1, x2, y2) in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

# Labels
plt.text(0.5, 0.95, "Model Framework Overview", ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PROJ_HOME, "model_diagram.png"))