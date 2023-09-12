import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Your code to load data
data_dict = torch.load(
    '/scratch/zczqyc4/neighbor_structure/sdpa_identity_non_True_False_1_1.pth')
index = data_dict.get("index", None)
weight = data_dict.get("weight", None)

edge_tensor = np.array(index)

# Create a new NetworkX graph
G = nx.Graph()

# Add edges to the graph
for i, neighbors in enumerate(edge_tensor):
    for j, neighbor in enumerate(neighbors):
        if weight[i][j] != 0:
            G.add_edge(i, neighbor)

# Draw the graph
plt.figure(figsize=(12, 12))

# Use the spring layout algorithm to layout the graph
pos = nx.spring_layout(G)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color='skyblue',
    node_size=700,
    font_size=16,
    font_color='black',
    font_weight='bold',
    edge_color='gray'
)

plt.title("Graph Visualization from Tensor", fontsize=20)
plt.axis("off")  # Turn off the axis

# Save as high-res PNG or as PDF for publication-quality
plt.savefig("beautiful_graph.png", format="PNG")
# plt.savefig("beautiful_graph.pdf", format="PDF")

plt.show()
