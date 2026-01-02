import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import networkx as nx
import math
import os
from concurrent.futures import ProcessPoolExecutor
import time

# use proper seeding call
random.seed(42)
np.random.seed(42)

# limit BLAS threads to avoid oversubscription when using many processes
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Try to import GPU-accelerated libraries
try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    GPU_AVAILABLE = True
    print("RAPIDS cuML available - using GPU k-means")
except ImportError:
    GPU_AVAILABLE = False
    print("RAPIDS cuML not available - falling back to CPU k-means")


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]

    max_len = raw_lines[0]
    nodes = [ln.split() for ln in raw_lines[2:]]  # id x, y, centroid

    return nodes, max_len


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def nearest_neighbor_from_tuples(pts):
    """Nearest neighbor heuristic for pts list of (id,x,y) tuples."""
    if not pts:
        return []
    unvisited = pts.copy()
    start = unvisited[0]
    path = [start]
    unvisited.remove(start)

    while unvisited:
        last = path[-1]
        next_node = min(unvisited, key=lambda p: euclidean_distance((last[1], last[2]), (p[1], p[2])))
        path.append(next_node)
        unvisited.remove(next_node)

    return [(p[0], (p[1], p[2])) for p in path]


def calculate_total_distance(path):
    total_distance = 0.0
    for i in range(len(path) - 1):
        total_distance += euclidean_distance(path[i][1], path[i + 1][1])
    return total_distance


def opt(path):
    """Improve the given path using the 2-opt algorithm."""
    improved = True
    best_path = path[:]
    best_distance = calculate_total_distance(best_path)

    while improved:
        improved = False
        for i in range(1, len(best_path) - 1):
            for j in range(i + 1, len(best_path)):
                if j - i == 1:
                    continue  # Skip adjacent nodes
                new_path = best_path[:]
                new_path[i:j] = reversed(best_path[i:j])
                new_distance = calculate_total_distance(new_path)
                if new_distance < best_distance:
                    best_path = new_path
                    best_distance = new_distance
                    improved = True

    return best_path


# Top-level helper (picklable) used by ProcessPoolExecutor
def process_cluster(cluster):
    """Compute NN tour and 2-opt improvement for a single cluster."""
    if not cluster:
        return None
    tour = nearest_neighbor_from_tuples(cluster)
    improved = opt(tour)
    return (tour, improved)


def k_means_cpu(nodes, k, iter=100):
    """Run k-means on CPU and return centroids and clusters where clusters contain tuples (id, x, y)."""
    # convert to tuples for numeric ops
    pts = [(int(node[0]), float(node[1]), float(node[2])) for node in nodes]

    # Initialize centroids randomly from existing points
    centroids = random.sample([(p[1], p[2]) for p in pts], k)

    clusters = []
    for _ in range(iter):
        clusters = [[] for _ in range(k)]

        # Assign points to nearest centroid
        for p in pts:
            distances = [np.linalg.norm(np.array((p[1], p[2])) - np.array(c)) for c in centroids]
            idx = int(np.argmin(distances))
            clusters[idx].append(p)

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


def k_means_gpu(nodes, k):
    """Run k-means on GPU using RAPIDS cuML and return clusters as list of (id, x, y) tuples."""
    import cupy as cp

    # Build position array and node list
    positions = np.array([[float(n[1]), float(n[2])] for n in nodes], dtype=np.float32)
    pts = [(int(node[0]), float(node[1]), float(node[2])) for node in nodes]

    # Move to GPU and run k-means
    positions_gpu = cp.asarray(positions)
    kmeans = cuKMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(positions_gpu)
    labels = cp.asnumpy(kmeans.labels_)

    # Build clusters on CPU
    clusters = [[] for _ in range(k)]
    for idx, lbl in enumerate(labels):
        clusters[int(lbl)].append(pts[idx])

    # Return dummy centroids (not used in this pipeline)
    centroids = []
    return centroids, clusters


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

    plt.close()  # Close to free memory


if __name__ == "__main__":
    # Process all robot files roboter01.txt .. roboter11.txt and save before/after images
    base_dir = os.path.dirname(__file__)
    candidate_dirs = ["A2_Gießroboter", "A2_Gieroboter", "A2_Gierobotter", "A2_Gieroboter"]

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

        print(f"\n{'='*60}")
        print(f"Processing {fname} (path: {fpath})")
        print(f"{'='*60}")

        t0 = time.time()
        nodes, max_len = read_file(fpath)

        k = 3  # Number of clusters

        # Use GPU k-means if available, else CPU
        t_km_start = time.time()
        if GPU_AVAILABLE:
            try:
                print(f"Running k-means on GPU (k={k})...")
                centroids, clusters = k_means_gpu(nodes, k)
            except Exception as e:
                print(f"GPU k-means failed ({e}), falling back to CPU")
                centroids, clusters = k_means_cpu(nodes, k)
        else:
            print(f"Running k-means on CPU (k={k})...")
            centroids, clusters = k_means_cpu(nodes, k)
        t_km_end = time.time()
        print(f"K-means took {t_km_end - t_km_start:.3f}s")

        # Parallel process non-empty clusters across CPU cores
        non_empty = [c for c in clusters if c]
        workers = min(len(non_empty) or 1, os.cpu_count() or 1)
        print(f"Processing {len(non_empty)} non-empty clusters with {workers} worker(s)...")

        t_par_start = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(process_cluster, non_empty))
        t_par_end = time.time()
        print(f"Parallel cluster processing took {t_par_end - t_par_start:.3f}s")

        # Unpack results
        cluster_tours = [r[0] for r in results if r]
        improved_tours = [r[1] for r in results if r]

        # Save images named after the robot file
        before_path = os.path.join(base_dir, f"{os.path.splitext(fname)[0]}_cluster_tours_before.png")
        visualise_all_paths(cluster_tours, title=f'{fname} - Cluster-local NN tours', save_path=before_path)

        after_path = os.path.join(base_dir, f"{os.path.splitext(fname)[0]}_cluster_tours_after_2opt.png")
        visualise_all_paths(improved_tours, title=f'{fname} - Cluster-local improved (2-opt) tours', save_path=after_path)

        # Print cluster distances
        print("\nDistance before improvement:")
        for idx, tour in enumerate(cluster_tours):
            dist = calculate_total_distance(tour)
            print(f" Cluster {idx + 1}: {dist:.2f}")
        print("Distance after improvement:")
        for idx, tour in enumerate(improved_tours):
            dist = calculate_total_distance(tour)
            print(f" Cluster {idx + 1}: {dist:.2f}")

        t_end = time.time()
        print(f"Total time for {fname}: {t_end - t0:.3f}s")

    print('\n' + '='*60)
    print('Processing complete.')
    print('='*60)

