import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import networkx as nx
import math
import os

# use proper seeding call
random.seed(42)


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]

    max_len = raw_lines[0]
    nodes = [ln.split() for ln in raw_lines[2:]]  # id x, y, centroid

    return nodes, max_len


def plot_graph(nodes):
    x = [float(node[1]) for node in nodes]
    y = [float(node[2]) for node in nodes]

    plt.scatter(x, y)

    for node in nodes:
        plt.text(float(node[1]), float(node[2]), node[0])

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Graph Nodes')
    plt.grid(True)
    plt.show()


def k_means(nodes, k, iter=100):
    """Run k-means and return centroids and clusters where clusters contain tuples (id, x, y).

    nodes: list of [id, x, y, ...]
    """
    # convert to tuples for numeric ops
    pts = [(int(node[0]), float(node[1]), float(node[2])) for node in nodes]

    # Initialize centroids randomly from existing points
    centroids = random.sample([(p[1], p[2]) for p in pts], k)

    clusters = []
    for _ in range(iter):
        clusters = [[] for _ in range(k)]

        # Assign points to nearest centroid (vectorized)
        pts_array = np.array([(p[1], p[2]) for p in pts])
        centroids_array = np.array(centroids)

        for idx, p in enumerate(pts):
            # Use squared distance (avoids sqrt for faster computation)
            distances = np.sum((pts_array[idx:idx + 1] - centroids_array) ** 2, axis=1)
            nearest_centroid = int(np.argmin(distances))
            clusters[nearest_centroid].append(p)

        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = (
                    sum(pt[1] for pt in cluster) / len(cluster),
                    sum(pt[2] for pt in cluster) / len(cluster)
                )
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.choice([(p[1], p[2]) for p in pts]))

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return centroids, clusters


def plot_graph_with_clusters(nodes, centroids, clusters):
    colors = sns.color_palette("hsv", max(1, len(clusters)))

    for idx, cluster in enumerate(clusters):
        # cluster contains tuples (id, x, y)
        x = [point[1] for point in cluster]
        y = [point[2] for point in cluster]
        plt.scatter(x, y, color=colors[idx], label=f'Cluster {idx + 1}')

    # Plot centroids
    centroid_x = [centroid[0] for centroid in centroids]
    centroid_y = [centroid[1] for centroid in centroids]
    plt.scatter(centroid_x, centroid_y, color='black', marker='X', s=100, label='Centroids')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Graph Nodes with Clusters')
    plt.legend()
    plt.grid(True)
    plt.show()


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def nearest_neighbor(nodes):
    # Convert nodes to a list of tuples: (id:int, x:float, y:float)
    pts = [(int(node[0]), float(node[1]), float(node[2])) for node in nodes]

    if not pts:
        return []

    unvisited = set(range(len(pts)))
    start_idx = 0
    path = [pts[start_idx]]
    unvisited.remove(start_idx)

    while unvisited:
        last = path[-1]
        # choose the unvisited point with minimum euclidean distance to last
        next_idx = min(unvisited, key=lambda idx: euclidean_distance((last[1], last[2]), (pts[idx][1], pts[idx][2])))
        path.append(pts[next_idx])
        unvisited.remove(next_idx)

    # return list of (id, (x, y)) tuples
    return [(p[0], (p[1], p[2])) for p in path]


def nearest_neighbor_from_tuples(pts):
    """Nearest neighbor heuristic for pts list of (id,x,y) tuples."""
    if not pts:
        return []
    unvisited = set(range(len(pts)))  # Use indices instead of removing from list
    start_idx = 0
    path = [pts[start_idx]]
    unvisited.remove(start_idx)

    while unvisited:
        last = path[-1]
        # Find unvisited point with minimum distance
        next_idx = min(unvisited, key=lambda idx: euclidean_distance((last[1], last[2]), (pts[idx][1], pts[idx][2])))
        path.append(pts[next_idx])
        unvisited.remove(next_idx)

    return [(p[0], (p[1], p[2])) for p in path]


def visualise_path(path):
    """Visualise a single path given as a list of (id, (x, y)) using networkx and seaborn.
    Kept for backward compatibility.
    """
    if not path:
        print("Empty path, nothing to visualise")
        return

    G = nx.Graph()

    # nodes as strings (labels)
    node_labels = [str(n[0]) for n in path]
    # edges between consecutive nodes
    edges = [(node_labels[i], node_labels[i + 1]) for i in range(len(node_labels) - 1)]

    G.add_nodes_from(node_labels)
    G.add_edges_from(edges)

    # position mapping from node label to coordinate
    pos = {str(n[0]): (n[1][0], n[1][1]) for n in path}

    sns.set_theme(style="white")

    nx.draw(
        G, pos,
        with_labels=True,
        node_color=[sns.color_palette("deep")[0]] * len(node_labels),
        node_size=800,
        edge_color="gray"
    )

    plt.title('Nearest Neighbor Path')
    plt.show()


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

    sns.set_theme(style="white")
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


def calculate_total_distance(path):
    total_distance = 0.0
    for i in range(len(path) - 1):
        total_distance += euclidean_distance(path[i][1], path[i + 1][1])
    return total_distance


def opt(path):
    """Improve the given path using the 2-opt algorithm."""
    if len(path) <= 3:
        return path

    # Pre-calculate all pairwise distances
    n = len(path)
    distances = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(path[i][1], path[j][1])
            distances[i][j] = dist
            distances[j][i] = dist

    # Calculate initial tour distance
    def tour_distance(path_indices):
        total = 0.0
        for i in range(len(path_indices) - 1):
            total += distances[path_indices[i]][path_indices[i + 1]]
        return total

    improved = True
    best_order = list(range(n))  # Track order instead of reordering path
    best_distance = tour_distance(best_order)

    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Calculate change in distance for 2-opt swap
                old_dist = (distances[best_order[i - 1]][best_order[i]] +
                            distances[best_order[j]][best_order[(j + 1) % n]])
                new_dist = (distances[best_order[i - 1]][best_order[j]] +
                            distances[best_order[i]][best_order[(j + 1) % n]])

                if new_dist < old_dist:
                    # Reverse the segment between i and j
                    best_order[i:j + 1] = reversed(best_order[i:j + 1])
                    best_distance = tour_distance(best_order)
                    improved = True
                    break
            if improved:
                break

    # Reorder path according to best_order
    return [path[i] for i in best_order]


if __name__ == "__main__":
    # Process all robot files roboter01.txt .. roboter11.txt and save before/after images
    base_dir = os.path.dirname(__file__)
    candidate_dirs = ["A2_Gießroboter", "A2_Gieroboter", "A2_Gierobotter", "A2_Gieroboter"]

    # Store results for all robots
    results = []

    for i in range(1, 12):
        fname = f"roboter{i:02d}.txt"
        fpath = None
        for d in candidate_dirs:
            p = os.path.join(base_dir, d, fname)
            if os.path.exists(p):
                fpath = p
                break
        if fpath is None:
            print(f"Skipping {fname}: not found in candidate dirs")
            continue

        print(f"\nProcessing {fname} (path: {fpath})")
        nodes, max_len = read_file(fpath)
        max_len = float(max_len)
        print(f"max_len = {max_len}")

        # Start with k=1 and increase until longest path < max_len
        k = 1
        found = False
        while not found:
            print(f"\n  Trying k = {k}...")
            centroids, clusters = k_means(nodes, k)

            # For each cluster compute a local tour (nearest neighbour) and then 2-opt improve it
            cluster_tours = []
            improved_tours = []
            for cluster in clusters:
                if len(cluster) == 0:
                    continue
                # nearest_neighbor_from_tuples expects list of (id,x,y)
                tour = nearest_neighbor_from_tuples(cluster)
                improved = opt(tour)
                cluster_tours.append(tour)
                improved_tours.append(improved)

            # Calculate distances
            tour_distances = []
            for tour in improved_tours:
                dist = calculate_total_distance(tour)
                tour_distances.append(dist)

            # Find longest path
            max_distance = max(tour_distances) if tour_distances else 0
            print(f"  Longest path distance: {max_distance:.2f}")

            # Check if condition is met
            if max_distance < max_len:
                found = True
                print(f"  ✓ Condition met! Longest path ({max_distance:.2f}) < max_len ({max_len})")

                # Save images only when condition is met
                before_path = os.path.join(base_dir, f"{os.path.splitext(fname)[0]}_cluster_tours_before.png")
                visualise_all_paths(cluster_tours, title=f'{fname} - Cluster-local NN tours (k={k})',
                                    save_path=before_path)

                after_path = os.path.join(base_dir, f"{os.path.splitext(fname)[0]}_cluster_tours_after_2opt.png")
                visualise_all_paths(improved_tours, title=f'{fname} - Cluster-local improved (2-opt) tours (k={k})',
                                    save_path=after_path)

                # Print cluster distances
                print("  distance before improvement:")
                for idx, tour in enumerate(cluster_tours):
                    dist = calculate_total_distance(tour)
                    print(f"    Cluster {idx + 1}: {dist:.2f}")
                print("  distance after improvement:")
                for idx, tour in enumerate(improved_tours):
                    dist = calculate_total_distance(tour)
                    print(f"    Cluster {idx + 1}: {dist:.2f}")

                # Store result
                results.append((fname, k, max_distance))
            else:
                k += 1

    # Save all k values to a text file
    output_file = os.path.join(base_dir, "k_values.txt")
    with open(output_file, "w") as f:
        f.write("Robot File\tk (clusters needed)\tMax Distance\n")
        f.write("-" * 50 + "\n")
        for robot_file, k_value, max_dist in results:
            f.write(f"{robot_file}\t{k_value}\t{max_dist:.2f}\n")

    print(f'\nProcessing complete. K values saved to: {output_file}')
