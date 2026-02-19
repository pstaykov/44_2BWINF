import math
import numpy as np

np.random.seed(42)


def read_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]

    max_len = float(raw_lines[0])

    nodes = []
    for ln in raw_lines[2:]:
        parts = ln.split()
        if len(parts) < 3:
            continue

        x = int(parts[1])
        y = int(parts[2])
        nodes.append((x, y))

    return nodes, max_len


def euclidean_dist(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_route_length(route) -> float:
    total_dist = 0.0
    for i in range(len(route) - 1):
        total_dist += euclidean_dist(route[i], route[i + 1])
    return total_dist


def kmeans(points, k, n_iter=100):
    coords = np.asarray(points, dtype=float)
    centroids = coords[np.random.choice(coords.shape[0], k, replace=False)]

    for _ in range(n_iter):
        distances = np.array([[euclidean_dist(p, c) for c in centroids] for p in coords])
        closest_indices = np.argmin(distances, axis=1)

        new_centroids = []
        for i in range(k):
            cluster_coords = coords[closest_indices == i]
            if len(cluster_coords) > 0:
                new_centroids.append(cluster_coords.mean(axis=0))
            else:
                new_centroids.append(coords[np.random.choice(coords.shape[0])])

        new_centroids = np.array(new_centroids)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    clusters = [[] for _ in range(k)]
    for idx, cluster_id in enumerate(closest_indices):
        clusters[cluster_id].append(tuple(coords[idx]))

    return clusters, centroids


def solve_tsp(clust):
    if not clust:
        return [], 0.0
    if len(clust) == 1:
        return [clust[0]], 0.0

    unvisited = clust.copy()
    path = [unvisited.pop(0)]

    while unvisited:
        last = path[-1]
        next_idx = min(range(len(unvisited)), key=lambda i: euclidean_dist(last, unvisited[i]))
        path.append(unvisited.pop(next_idx))

    length = calculate_route_length(path)
    return path, length


def optimize_2opt(path, max_iterations=100):
    if len(path) < 4:
        return path[:]

    best_path = path[:]
    for _ in range(max_iterations):
        improved = False
        for i in range(1, len(best_path) - 2):
            for j in range(i + 1, len(best_path) - 1):
                old_dist = (euclidean_dist(best_path[i - 1], best_path[i]) +
                            euclidean_dist(best_path[j], best_path[j + 1]))
                new_dist = (euclidean_dist(best_path[i - 1], best_path[j]) +
                            euclidean_dist(best_path[i], best_path[j + 1]))

                if new_dist < old_dist:
                    best_path[i:j + 1] = best_path[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return best_path


def route_max_len_for_k(k: int, nodes, max_len: float) -> float:
    """
    Compute d(k): the maximum route length across all clusters for given k.
    Early-exits if the max already exceeds max_len.
    """
    clusters, _ = kmeans(nodes, k)

    worst = 0.0
    for clust in clusters:
        path, _ = solve_tsp(clust)
        path = optimize_2opt(path)
        length = calculate_route_length(path)
        worst = max(worst, length)
        if worst > max_len:
            return worst
    return worst


def k_works(k, nodes, max_len):
    return route_max_len_for_k(k, nodes, max_len) <= max_len


def find_optimal_k(nodes, max_len):
    """
    Implements:
      1) Interpolation-based growth of k using a local linear model between (k_prev, d_prev) and (k_curr, d_curr)
         until d(k_curr) <= L_max.
      2) Binary search on [k_start, k_curr] to find the smallest k* with d(k*) <= L_max.
    """
    n = len(nodes)
    if n == 0:
        return 0
    if n == 1:
        return 1

    # find an upper bound k_curr that works (d(k_curr) <= max_len)
    k_prev = 3
    d_prev = route_max_len_for_k(k_prev, nodes, max_len)

    if d_prev <= max_len:
        print(f"K={k_prev} works")
        return k_prev

    # Second point to be able to estimate slope
    k_curr = 4
    d_curr = route_max_len_for_k(k_curr, nodes, max_len)

    # Safety against pathological cases
    max_steps = 50
    steps = 0

    while d_curr > max_len and k_curr < n and steps < max_steps:
        steps += 1

        # Estimate slope a from two consecutive measurements
        denom = (k_curr - k_prev)
        a = (d_curr - d_prev) / denom if denom != 0 else 0.0

        # Compute interpolated next k
        if a < 0:
            # d decreases with k (expected). Compute k_new
            k_float = k_curr + (max_len - d_curr) / a
            k_new = int(math.ceil(k_float))
        else:
            # If the model says "not decreasing", fall back to a robust step
            k_new = min(n, max(k_curr + 1, k_curr * 2))

        # Ensure progress and stay in bounds
        k_new = max(k_new, k_curr + 1)
        k_new = min(k_new, n)

        k_prev, d_prev = k_curr, d_curr
        k_curr = k_new
        d_curr = route_max_len_for_k(k_curr, nodes, max_len)

    if d_curr > max_len:
        # No feasible k found up to n
        return None

    k_start = k_prev
    lo = k_start + 1
    hi = k_curr

    # binary search for minimal k that works
    while lo < hi:
        mid = (lo + hi) // 2
        d_mid = route_max_len_for_k(mid, nodes, max_len)
        if d_mid <= max_len:
            hi = mid
        else:
            lo = mid + 1

    print(f"K={lo} works (minimal)")
    return lo


if __name__ == "__main__":
    nodes, max_len = read_file("A2_Gießroboter/roboter11.txt")
    find_optimal_k(nodes, max_len)