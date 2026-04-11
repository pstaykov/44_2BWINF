import math
import numpy as np
import os

np.random.seed(42)


def read_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f.readlines() if ln.strip() != ""]
    coord_start = None
    for i, ln in enumerate(raw_lines):
        parts = ln.split()
        if len(parts) >= 3:
            coord_start = i
            break

    if coord_start is None:
        raise ValueError("No coordinate lines found in input file")
    if coord_start == 0:
        max_len = float(raw_lines[0])
        start_idx = 1
    else:
        max_len = float(raw_lines[coord_start - 1])
        start_idx = coord_start

    nodes = []  # list of (id, x, y)
    for ln in raw_lines[start_idx:]:
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            id_ = int(parts[0])
            x = int(parts[1])
            y = int(parts[2])
        except ValueError:
            continue
        nodes.append((id_, x, y))

    return nodes, max_len


def euclidean_dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def kmeans(points, k, n_iter=100):
    """
    points: list of (x,y) tuples or numpy array shape (n,2)
    returns: clusters as list of lists of point indices, centroids as numpy array (k,2)
    """
    coords = np.asarray(points, dtype=float)
    n = coords.shape[0]
    if k <= 0:
        return [[] for _ in range(k)], np.zeros((0, 2))
    if k > n:
        k = n

    centroids = coords[np.random.choice(coords.shape[0], k, replace=False)]

    for _ in range(n_iter):
        distances = np.sqrt(((coords[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
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
        clusters[cluster_id].append(int(idx))

    return clusters, centroids


def solve_tsp_indices(indices, coords):
    """
    Nearest-neighbor heuristic starting from the first index in indices.
    coords: list-like of (x,y) for all points; indices: list of integer indices of the points in this cluster.
    Returns ordered list of indices and total length.
    """
    if not indices:
        return [], 0.0
    if len(indices) == 1:
        return [indices[0]], 0.0

    unvisited = indices.copy()
    path = [unvisited.pop(0)]

    while unvisited:
        last = coords[path[-1]]
        next_idx = min(range(len(unvisited)), key=lambda i: euclidean_dist(last, coords[unvisited[i]]))
        path.append(unvisited.pop(next_idx))

    length = 0.0
    for i in range(len(path) - 1):
        length += euclidean_dist(coords[path[i]], coords[path[i + 1]])
    return path, length


def optimize_2opt_indices(path_indices, coords, max_iterations=100):
    if len(path_indices) < 4:
        return path_indices[:]

    best_path = path_indices[:]
    for _ in range(max_iterations):
        improved = False
        for i in range(1, len(best_path) - 2):
            for j in range(i + 1, len(best_path) - 1):
                a = coords[best_path[i - 1]]
                b = coords[best_path[i]]
                c = coords[best_path[j]]
                d = coords[best_path[j + 1]]
                old_dist = euclidean_dist(a, b) + euclidean_dist(c, d)
                new_dist = euclidean_dist(a, c) + euclidean_dist(b, d)

                if new_dist < old_dist:
                    best_path[i:j + 1] = best_path[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return best_path


def route_max_len_for_k(k, nodes, max_len):
    """
    Compute d(k): the maximum route length across all clusters for given k.
    Early-exits if the max already exceeds max_len.
    nodes: list of (id,x,y)
    """
    coords = [(x, y) for (_id, x, y) in nodes]
    clusters, _ = kmeans(coords, k)

    worst = 0.0
    for clust in clusters:
        path, _ = solve_tsp_indices(clust, coords)
        path = optimize_2opt_indices(path, coords)
        length = 0.0
        for i in range(len(path) - 1):
            length += euclidean_dist(coords[path[i]], coords[path[i + 1]])
        worst = max(worst, length)
        if worst > max_len:
            return worst
    return worst


def k_works(k, nodes, max_len):
    return route_max_len_for_k(k, nodes, max_len) <= max_len


def find_optimal_k(nodes, max_len):
    """
    Interpolation-based growth of k using a local linear model between (k_prev, d_prev) and (k_curr, d_curr)
    until d(k_curr) <= L_max.
    Binary search on [k_start, k_curr] to find the smallest k* with d(k*) <= L_max.
    Returns minimal k or None if not found.
    """
    n = len(nodes)
    if n == 0:
        return 0
    if n == 1:
        return 1

    # find an upper bound k_curr
    k_prev = 3
    d_prev = route_max_len_for_k(k_prev, nodes, max_len)

    if d_prev <= max_len:
        print(f"K={k_prev} works")
        return k_prev

    k_curr = 4
    d_curr = route_max_len_for_k(k_curr, nodes, max_len)

    max_steps = 50
    steps = 0

    while d_curr > max_len and k_curr < n and steps < max_steps:
        steps += 1

        denom = (k_curr - k_prev)
        a = (d_curr - d_prev) / denom if denom != 0 else 0.0

        # Compute interpolated next k
        if a < 0:
            k_float = k_curr + (max_len - d_curr) / a
            k_new = int(math.ceil(k_float))
        else:
            # fallback to robust step
            k_new = min(n, max(k_curr + 1, k_curr * 2))

        k_new = max(k_new, k_curr + 1)
        k_new = min(k_new, n)

        k_prev, d_prev = k_curr, d_curr
        k_curr = k_new
        d_curr = route_max_len_for_k(k_curr, nodes, max_len)

    if d_curr > max_len:
        return None

    k_start = k_prev
    lo = k_start + 1
    hi = k_curr

    while lo < hi:
        mid = (lo + hi) // 2
        d_mid = route_max_len_for_k(mid, nodes, max_len)
        if d_mid <= max_len:
            hi = mid
        else:
            lo = mid + 1

    print(f"K={lo} works (minimal)")
    return lo


def build_routes_for_k(nodes, k):
    """
    Return list of routes per cluster for given k.
    Each route is a dict: { 'ids': [id,...], 'coords': [(x,y),...], 'length': float, 'station': (x,y) }
    Station is the first tree in the route.
    """
    coords = [(x, y) for (_id, x, y) in nodes]
    id_list = [_id for (_id, x, y) in nodes]
    clusters, centroids = kmeans(coords, k)

    routes = []
    for ci, clust in enumerate(clusters):
        if not clust:
            routes.append({'ids': [], 'coords': [], 'length': 0.0,
                           'station': (int(round(centroids[ci][0])), int(round(centroids[ci][1])))})
            continue
        path, _ = solve_tsp_indices(clust, coords)
        path = optimize_2opt_indices(path, coords)
        route_coords = [coords[idx] for idx in path]
        route_ids = [id_list[idx] for idx in path]
        length = 0.0
        for i in range(len(route_coords) - 1):
            length += euclidean_dist(route_coords[i], route_coords[i + 1])

        # Ladestation ist die Koordinate des ersten Baums auf der Route
        station = (int(route_coords[0][0]), int(route_coords[0][1]))
        routes.append({'ids': route_ids, 'coords': route_coords, 'length': length, 'station': station})

    return routes


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "A2_Gießroboter")
    outputs_dir = os.path.join(script_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
    else:
        files = sorted(os.listdir(input_dir))
        txt_files = [f for f in files if f.lower().endswith('.txt')]

        if not txt_files:
            print(f"No .txt input files found in {input_dir}")

        for fname in txt_files:
            if fname.endswith('_out.txt'):
                continue

            input_path = os.path.join(input_dir, fname)
            print(f"Processing {input_path}...")
            try:
                nodes, max_len = read_file(input_path)
            except Exception as e:
                print(f"  Skipping {fname}: failed to read file: {e}")
                continue

            try:
                k = find_optimal_k(nodes, max_len)
                if k is None:
                    print(f"  Could not find feasible number of robots for {fname}")
                    out_name = fname + "_out.txt"
                    out_path = os.path.join(outputs_dir, out_name)
                    with open(out_path, 'w', encoding='utf-8') as out:
                        out.write("0\n")
                        if float(max_len).is_integer():
                            out.write(str(int(max_len)) + "\n")
                        else:
                            out.write(str(max_len) + "\n")
                    print(f"  Wrote failure output to {out_path}")
                    continue

                routes = build_routes_for_k(nodes, k)

                out_name = fname + "_out.txt"
                out_path = os.path.join(outputs_dir, out_name)

                with open(out_path, "w", encoding="utf-8") as out:
                    out.write(str(k) + "\n")
                    if float(max_len).is_integer():
                        out.write(str(int(max_len)) + "\n")
                    else:
                        out.write(str(max_len) + "\n")

                    for r in routes:
                        out.write(str(int(math.ceil(r['length']))) + "\n")
                        out.write(f"{r['station'][0]} {r['station'][1]}\n")
                        if r['ids']:
                            out.write(" ".join(str(i) for i in r['ids']) + "\n")
                        else:
                            out.write("\n")

                print(f"  Wrote output to {out_path}")

            except Exception as e:
                print(f"  Error processing {fname}: {e}")
                continue