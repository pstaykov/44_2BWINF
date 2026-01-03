import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import networkx as nx
import math
import os
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

random.seed(42)
np.random.seed(42)


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
        """Run k-means with vectorized operations for better performance."""
        pts = [(int(node[0]), float(node[1]), float(node[2])) for node in nodes]
        pts_array = np.array([(p[1], p[2]) for p in pts])

        # Initialize centroids randomly
        centroid_indices = np.random.choice(len(pts), k, replace=False)
        centroids = pts_array[centroid_indices].copy()

        for _ in range(iter):
            # Vectorized distance calculation
            distances = np.linalg.norm(pts_array[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([pts_array[assignments == j].mean(axis=0) if np.any(assignments == j)
                                       else pts_array[np.random.choice(len(pts_array))] for j in range(k)])

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        # Reformat clusters as before
        clusters = [[(pts[i][0], pts[i][1], pts[i][2]) for i in range(len(pts)) if assignments[i] == j]
                     for j in range(k)]
        return centroids.tolist(), [c for c in clusters if c]


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
    """Optimized nearest neighbor using scipy."""
    if len(pts) <= 1:
        return [(p[0], (p[1], p[2])) for p in pts]

    # Convert to list of coordinates for easier indexing
    coords = np.array([(p[1], p[2]) for p in pts], dtype=np.float64)
    unvisited = set(range(len(pts)))
    path = [pts[0]]
    unvisited.remove(0)

    while unvisited:
        # Get last coordinate as 2D array (1, 2)
        last_point = path[-1]
        last_coord = np.array([[last_point[1], last_point[2]]], dtype=np.float64)

        # Get unvisited coordinates
        unvisited_indices = np.array(list(unvisited), dtype=np.int64)
        unvisited_coords = coords[unvisited_indices]

        # Calculate distances
        distances = cdist(last_coord, unvisited_coords)[0]

        # Find nearest
        nearest_idx = np.argmin(distances)
        next_global_idx = unvisited_indices[nearest_idx]

        path.append(pts[next_global_idx])
        unvisited.remove(next_global_idx)

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


def calculate_total_distance(path):
    """Vectorized distance calculation."""
    if len(path) <= 1:
        return 0.0
    coords = np.array([p[1] for p in path])
    diffs = np.diff(coords, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)


def opt(path):
    """Improve the given path using the 2-opt algorithm with vectorized operations."""
    if len(path) <= 3:
        return path

    n = len(path)

    # Skip 2-opt for very large clusters to save computation time
    if n > 500:
        return path

    # Pre-calculate all pairwise distances using scipy
    coords = np.array([p[1] for p in path])
    distances = cdist(coords, coords)

    # Calculate initial tour distance
    def tour_distance(order):
        total = 0.0
        for i in range(len(order) - 1):
            total += distances[order[i]][order[i + 1]]
        return total

    improved = True
    best_order = list(range(n))
    best_distance = tour_distance(best_order)

    # 2-opt with early termination for large paths
    max_iterations = min(100, n)
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        # Vectorized distance comparisons
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Calculate change in distance for 2-opt swap
                a, b = best_order[i - 1], best_order[i]
                c, d = best_order[j], best_order[(j + 1) % n]

                old_dist = distances[a][b] + distances[c][d]
                new_dist = distances[a][c] + distances[b][d]

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


def process_single_cluster(cluster):
    """Process a single cluster: compute NN tour and apply 2-opt improvement.
    This function is designed to run in parallel."""
    if len(cluster) == 0:
        return None, None

    tour = nearest_neighbor_from_tuples(cluster)
    improved = opt(tour)
    dist = calculate_total_distance(improved)

    return tour, improved


if __name__ == "__main__":
    # Process all robot files roboter01.txt .. roboter11.txt and save before/after images
    base_dir = os.path.dirname(__file__)
    candidate_dirs = ["A2_Gießroboter", "A2_Gieroboter", "A2_Gierobotter", "A2_Gieroboter"]

    # Store results for all robots
    results = []

    # Determine number of worker processes
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} worker processes for parallel computation")

    for i in range(8, 12):
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
        num_nodes = len(nodes)
        print(f"max_len = {max_len}, num_nodes = {num_nodes}")

        # Use interpolation-based search assuming linear relationship between k and max_distance
        # Repeat until distance difference is 30 or less
        k = 1
        previous_k = None
        previous_distance = None
        found = False
        max_k = num_nodes  # Can't have more clusters than nodes

        while not found:
            print(f"\n  Trying k = {k}...")
            centroids, clusters = k_means(nodes, k)

            # Process clusters in parallel using ThreadPoolExecutor (better for I/O-bound tasks)
            # Use ProcessPoolExecutor for CPU-intensive 2-opt on very large clusters
            cluster_tours = []
            improved_tours = []

            # Use ThreadPoolExecutor for parallel processing of clusters
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results_iter = executor.map(process_single_cluster, clusters)
                for tour, improved in results_iter:
                    if tour is not None:
                        cluster_tours.append(tour)
                        improved_tours.append(improved)

            # Calculate distances
            tour_distances = []
            for tour in improved_tours:
                dist = calculate_total_distance(tour)
                tour_distances.append(dist)

            # Find longest path
            max_distance = max(tour_distances) if tour_distances else 0
            distance_diff = max_distance - max_len
            print(f"  Longest path distance: {max_distance:.2f} (diff from max_len: {distance_diff:.2f})")

            # Check if condition is met (distance within 30 units of max_len)
            if max_distance <= max_len:
                found = True
                print(f"  ✓ Condition met! Longest path ({max_distance:.2f}) smaller than max_len ({max_len})")

                # Visualize clusters with centroids
                visualise_clusters_with_centroids(clusters, centroids, title=f'{fname} - Clusters (k={k})')

                # Save images only when condition is met
                # before_path = os.path.join(base_dir, f"{os.path.splitext(fname)[0]}_cluster_tours_before.png")
                visualise_all_paths(cluster_tours, title=f'{fname} - Cluster-local NN tours (k={k})')
                                    # save_path=before_path)

                # after_path = os.path.join(base_dir, f"{os.path.splitext(fname)[0]}_cluster_tours_after_2opt.png")
                visualise_all_paths(improved_tours, title=f'{fname} - Cluster-local improved (2-opt) tours (k={k})')
                                    # save_path=after_path)

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
                # Store current state before calculating new k
                current_k = k
                current_distance = max_distance

                # Use linear interpolation to estimate next k
                if previous_k is not None and previous_distance is not None:
                    # Assume linear relationship: distance = a * k + b
                    # Calculate slope: (current_distance - previous_distance) / (current_k - previous_k)
                    delta_distance = current_distance - previous_distance
                    delta_k = current_k - previous_k

                    if abs(delta_distance) > 0.01 and delta_k != 0:  # Avoid division by zero
                        slope = delta_distance / delta_k  # negative slope (as k increases, distance decreases)

                        # If slope is very small (near-flat), we're converging slowly
                        # Use conservative increments instead of extrapolating
                        if abs(slope) < 0.1:  # Very flat slope
                            k = min(current_k + 10, max_k)  # Increment by 10, but don't exceed max_k
                            print(f"  Slope very small ({slope:.4f}), using conservative increment: next k = {k}")
                        else:
                            # Solve for k where distance = max_len
                            # max_len = current_distance + slope * (k - current_k)
                            # k = current_k + (max_len - current_distance) / slope
                            k_new = current_k + (max_len - current_distance) / slope
                            # Bound the result and ensure we increment by at least 1
                            k_bounded = max(current_k + 1, int(round(k_new)))
                            k = min(k_bounded, max_k)  # Ensure we don't exceed max_k

                            if k != k_bounded:
                                print(f"  Linear interpolation: slope={slope:.4f}, k would be {k_bounded} but capped at {k}")
                            else:
                                print(f"  Linear interpolation: slope={slope:.4f}, next k = {k}")
                    else:
                        k = min(current_k + 1, max_k)
                        print(f"  Distance change too small, incrementing k")
                else:
                    k = min(k + 1, max_k)

                # Store current values as previous for next iteration
                previous_k = current_k
                previous_distance = current_distance

    # Save all k values to a text file
    output_file = os.path.join(base_dir, "k_values.txt")
    with open(output_file, "w") as f:
        f.write("Robot File\tk (clusters needed)\tMax Distance\n")
        f.write("-" * 50 + "\n")
        for robot_file, k_value, max_dist in results:
            f.write(f"{robot_file}\t{k_value}\t{max_dist:.2f}\n")

    print(f'\nProcessing complete. K values saved to: {output_file}')
