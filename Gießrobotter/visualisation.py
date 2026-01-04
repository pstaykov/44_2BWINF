import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

sns.set_theme(style="white")


def visualise_all_paths(paths, title='Cluster-local tours', save_path=None):
    """Visualise multiple paths (one per cluster) in a single plot with different colors.

    paths: list of paths, where each path is list of (id, (x,y)) tuples
    save_path: optional file path to save the resulting image (PNG)
    """
    if not paths:
        print("No paths to visualise")
        return

    # Collect positions and nodes
    pos = {}
    for path in paths:
        for node in path:
            nid = str(node[0])
            if nid not in pos:
                pos[nid] = (node[1][0], node[1][1])

    # Build combined graph (nodes only needed, edges drawn per path)
    G = nx.Graph()
    G.add_nodes_from(pos.keys())

    colors = sns.color_palette("deep", len(paths))

    plt.figure()
    # draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=300)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # draw edges per path with a different color
    for idx, path in enumerate(paths):
        node_labels = [str(n[0]) for n in path]
        edges = [(node_labels[i], node_labels[i + 1]) for i in range(len(node_labels) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=[colors[idx]], width=2.5, alpha=0.9)

    plt.title(title)
    plt.axis('off')

    # Save the figure if requested
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualisation to: {save_path}")
        except Exception as e:
            print(f"Failed to save image to {save_path}: {e}")

    plt.show()


def visualise_clusters_with_centroids(clusters, centroids, title='Clusters with Centroids'):
    """Visualise clusters with their centroids.

    clusters: list of clusters, where each cluster contains tuples (id, x, y)
    centroids: list of centroid coordinates (x, y)
    """
    if not clusters:
        print("No clusters to visualise")
        return

    colors = sns.color_palette("hsv", len(clusters))

    # Plot each cluster with a different color
    for idx, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
        x = [point[1] for point in cluster]
        y = [point[2] for point in cluster]
        plt.scatter(x, y, color=colors[idx], label=f'Cluster {idx + 1}', s=50)

    # Plot centroids
    if centroids:
        centroid_x = [centroid[0] for centroid in centroids]
        centroid_y = [centroid[1] for centroid in centroids]
        plt.scatter(centroid_x, centroid_y, color='black', marker='X', s=200,
                   edgecolors='red', linewidths=2, label='Centroids', zorder=5)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

