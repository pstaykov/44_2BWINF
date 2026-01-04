import random
import numpy as np
import math
import os
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from Gießrobotter.visualisation import visualise_all_paths, visualise_clusters_with_centroids

random.seed(42)
np.random.seed(42)


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]

    max_len = raw_lines[0]
    nodes = [ln.split() for ln in raw_lines[2:]]  # id x, y, centroid

    return nodes, max_len


def k_means(nodes, k, iter=50):
        """Run k-means with vectorized operations for better performance."""
        pts = [(int(node[0]), float(node[1]), float(node[2])) for node in nodes]
        if len(pts) == 0:
            return [], []
        # Ensure k is within sensible bounds
        if k <= 0:
            k = 1
        if k > len(pts):
            k = len(pts)
        pts_array = np.array([(p[1], p[2]) for p in pts])

        # Initialize centroids randomly
        centroid_indices = np.random.choice(len(pts), k, replace=False)
        centroids = pts_array[centroid_indices].copy()

        # initialize assignments for static analyzers; will be overwritten in loop
        assignments = np.zeros(len(pts), dtype=int)
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
        next_global_idx = int(unvisited_indices[nearest_idx])

        path.append(pts[next_global_idx])
        unvisited.remove(next_global_idx)

    return [(p[0], (p[1], p[2])) for p in path]


def calculate_total_distance(path):
    """Vectorized distance calculation."""
    if len(path) <= 1:
        return 0.0
    coords = np.array([p[1] for p in path])
    diffs = np.diff(coords, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)


def opt(path):
    """2opt algorithm to improve a path by swapping edges."""
    if len(path) <= 3:
        return path

    n = len(path)

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


def evaluate_k(nodes, k, num_workers):
    """Run k-means for given k, process clusters (parallel), and return (max_distance, centroids, clusters, cluster_tours, improved_tours).

    Results are returned so callers can visualise the final chosen solution. Only the max_distance is used for decisions.
    """
    centroids, clusters = k_means(nodes, k)

    cluster_tours = []
    improved_tours = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results_iter = executor.map(process_single_cluster, clusters)
        for tour, improved in results_iter:
            if tour is not None:
                cluster_tours.append(tour)
                improved_tours.append(improved)

    # Compute distances for improved tours
    tour_distances = [calculate_total_distance(t) for t in improved_tours]
    max_distance = max(tour_distances) if tour_distances else 0.0

    return max_distance, centroids, clusters, cluster_tours, improved_tours


def find_min_k(nodes, max_len, num_nodes, num_workers):
    """Find the minimal k such that the longest cluster tour distance is within max_len.

    Uses an adaptive approach: starts with k = 1, increases k until the condition is met,
    then performs a binary search to find the minimal k.
    """
    # Use interpolation-based search assuming linear relationship between k and max_distance
    k = 1
    previous_k = None
    previous_distance = None
    # current k that satisfies the constraint (d(k) <= max_len) when found
    k_current = None
    found = False
    max_k = num_nodes  # Can't have more clusters than nodes

    # Keep track of tested k values and cache computed distances to avoid recomputation
    tested_ks = []
    computed = {}  # k -> (max_distance, centroids, clusters, cluster_tours, improved_tours)

    while not found:
        # Record tested k
        if k in tested_ks:
            # Avoid infinite loops when interpolation suggests same k
            k = min(k + 1, max_k)
        tested_ks.append(k)

        print(f"\n  Trying k = {k}...")

        # Evaluate k (use cache when possible)
        if k in computed:
            max_distance, centroids, clusters, cluster_tours, improved_tours = computed[k]
        else:
            max_distance, centroids, clusters, cluster_tours, improved_tours = evaluate_k(nodes, k, num_workers)
            computed[k] = (max_distance, centroids, clusters, cluster_tours, improved_tours)

        distance_diff = max_distance - max_len
        print(f"  Longest path distance: {max_distance:.2f} (diff from max_len: {distance_diff:.2f})")

        # Check if condition is met
        if max_distance <= max_len:
            found = True
            print(f"  ✓ Condition met for k = {k}: longest path ({max_distance:.2f}) <= max_len ({max_len})")
            k_current = k
            # Do not visualise yet; perform binary search to find minimal k
            break
        else:
            # Store current state before calculating new k
            current_k = k
            current_distance = max_distance

            # Use linear interpolation to estimate next k
            if previous_k is not None and previous_distance is not None:
                delta_distance = current_distance - previous_distance
                delta_k = current_k - previous_k

                if abs(delta_distance) > 0.01 and delta_k != 0:
                    slope = delta_distance / delta_k

                    # If slope is non-negative or very small, use conservative increment
                    if slope >= 0 or abs(slope) < 0.1:
                        k = min(current_k + 10, max_k)
                        print(f"  Slope {slope:.4f} not suitable for extrapolation, using conservative increment: next k = {k}")
                    else:
                        # Solve for k where distance = max_len: max_len = current_distance + slope * (k - current_k)
                        k_new = current_k + (max_len - current_distance) / slope
                        # Round up to be safe (we need k such that distance <= max_len)
                        k_bounded = max(current_k + 1, int(math.ceil(k_new)))
                        k = min(k_bounded, max_k)

                        if k != k_bounded:
                            print(f"  Linear interpolation: slope={slope:.4f}, k would be {k_bounded} but capped at {k}")
                        else:
                            print(f"  Linear interpolation: slope={slope:.4f}, next k = {k}")
                else:
                    k = min(current_k + 1, max_k)
                    print(f"  Distance change too small, incrementing k to {k}")
            else:
                k = min(k + 1, max_k)

            # Store current values as previous for next iteration
            previous_k = current_k
            previous_distance = current_distance

        # Stop if we've reached max_k
        if k >= max_k:
            # Evaluate the final k if not already
            if k not in computed:
                computed[k] = evaluate_k(nodes, k, num_workers)
            # If still not satisfying, accept k = max_k
            max_distance = computed[k][0]
            if max_distance > max_len:
                print(f"  Reached k = max_k ({max_k}) but condition still not met (distance={max_distance:.2f}), accepting max_k")
            k_current = k
            found = True
            break

    # At this point k_current is a k such that d(k_current) <= max_len (or we've capped at max_k).
    # Now run the binary search to find the minimal k* in [k_start, k_current].

    # Fallback: ensure k_current is defined (static analyzers may not infer it's always set)
    if k_current is None:
        k_current = max(tested_ks) if tested_ks else max_k

    # Determine the largest already-tested k strictly less than k_current
    tested_less = [t for t in tested_ks if t < k_current]
    max_tested_less = max(tested_less) if tested_less else 0

    # Find biggest fraction of form 1/2^i such that k_start = floor(k_current * (1 - 1/2^i)) > max_tested_less
    k_start = max_tested_less + 1
    # Try increasing i until the k_start would be <= max_tested_less, then use previous
    i = 1
    last_valid = k_start
    while True:
        candidate = int(math.floor(k_current * (1 - 1 / (2 ** i))))
        if candidate <= max_tested_less or candidate < 1:
            break
        last_valid = candidate
        i += 1
    k_start = max(1, last_valid)

    # Ensure interval is valid
    if k_start >= k_current:
        k_start = max(1, k_current - 1)

    print(f"  Binary search interval: [{k_start}, {k_current}] (tested_ks: {sorted(set(tested_ks))})")

    # Binary search for minimal k
    low = k_start
    high = k_current

    best_k = k_current

    while low <= high:
        mid = (low + high) // 2
        if mid in computed and len(computed[mid]) == 5:
            mid_distance = computed[mid][0]
        else:
            # Capture all returned values to avoid NoneType errors later
            mid_res = evaluate_k(nodes, mid, num_workers)
            computed[mid] = mid_res
            mid_distance = mid_res[0]

        print(f"    Binary search testing k={mid} -> d={mid_distance:.2f}")

        if mid_distance <= max_len:
            best_k = mid
            high = mid - 1
        else:
            low = mid + 1

    # Best_k is minimal k* such that d(k*) <= max_len
    print(f"  Found minimal k* = {best_k} (d={computed[best_k][0]:.2f})")

    # Get artifacts for best_k (compute if missing)
    if best_k in computed and len(computed[best_k]) == 5:
        _, centroids, clusters, cluster_tours, improved_tours = computed[best_k]
        max_distance = computed[best_k][0]
    else:
        max_distance, centroids, clusters, cluster_tours, improved_tours = evaluate_k(nodes, best_k, num_workers)
        computed[best_k] = (max_distance, centroids, clusters, cluster_tours, improved_tours)

    return best_k, max_distance, centroids, clusters, cluster_tours, improved_tours


if __name__ == "__main__":
    # Process all robot files roboter01.txt .. roboter11.txt and save before/after images
    base_dir = os.path.dirname(__file__)
    candidate_dirs = ["A2_Gießroboter", "A2_Gieroboter", "A2_Gierobotter", "A2_Gieroboter"]

    # Store results for all robots
    results = []

    # Determine number of worker processes
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} worker processes for parallel computation")

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
        num_nodes = len(nodes)
        print(f"max_len = {max_len}, num_nodes = {num_nodes}")

        # Delegate finding minimal k to helper to keep main loop compact
        best_k, max_distance, centroids, clusters, cluster_tours, improved_tours = find_min_k(nodes, max_len, num_nodes, num_workers)

        # Visualise and store result for the minimal k*
        visualise_clusters_with_centroids(clusters, centroids, title=f'{fname} - Clusters (k={best_k})')
        visualise_all_paths(cluster_tours, title=f'{fname} - Cluster-local NN tours (k={best_k})')
        visualise_all_paths(improved_tours, title=f'{fname} - Cluster-local improved (2-opt) tours (k={best_k})')

        print("  distance before improvement:")
        for idx, tour in enumerate(cluster_tours):
            dist = calculate_total_distance(tour)
            print(f"    Cluster {idx + 1}: {dist:.2f}")
        print("  distance after improvement:")
        for idx, tour in enumerate(improved_tours):
            dist = calculate_total_distance(tour)
            print(f"    Cluster {idx + 1}: {dist:.2f}")

        # Store result
        results.append((fname, best_k, max_distance))

    # Save all k values to a text file
    output_file = os.path.join(base_dir, "k_values.txt")
    with open(output_file, "w") as f:
        f.write("Robot File\tk (clusters needed)\tMax Distance\n")
        f.write("-" * 50 + "\n")
        for robot_file, k_value, max_dist in results:
            f.write(f"{robot_file}\t{k_value}\t{max_dist:.2f}\n")

    print(f'\nProcessing complete. K values saved to: {output_file}')
