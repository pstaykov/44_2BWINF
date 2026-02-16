import math
import random
import numpy as np

random.seed(42)
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


def k_works(k, nodes, max_len):
    clusters, _ = kmeans(nodes, k)

    for clust in clusters:
        path, _ = solve_tsp(clust)
        path = optimize_2opt(path)
        length = calculate_route_length(path)
        if length > max_len:
            return False

    return True


def find_optimal_k(nodes, max_len):
    for k in range(2, len(nodes)):
        if k_works(k, nodes, max_len):
            print(f"K={k} works")
            return k
    return None


if __name__ == "__main__":
    nodes, max_len = read_file("A2_Gießroboter/roboter03.txt")
    find_optimal_k(nodes, max_len)