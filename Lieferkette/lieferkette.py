import os

def read_file(path):
    """Parse a delivery file and return (T, stages, Strassen_bidirectional, S_node, T_node).

    - T: int deadline
    - stages: dict mapping node (str) -> stage index (int)
    - Strassen_bidirectional: list of (u, v, w) with w as int
    - S_node / T_node: 'S' or 'T' if present otherwise None
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f.readlines() if ln.strip() != ""]

    if not raw:
        raise ValueError("Empty file")

    # Some sample files include a non-numeric first line (e.g. 'UNKNOWN'), handle that.
    idx = 0
    try:
        T = int(raw[idx])
    except ValueError:
        idx += 1
        T = int(raw[idx])
    idx += 1

    # Schritte (not used directly here, but keep parsing)
    Schritte = int(raw[idx])
    idx += 1

    WerkeSchritte = [int(x) for x in raw[idx].split()]
    idx += 1

    nWerke = int(raw[idx])
    idx += 1

    Werke_raw = [ln.split() for ln in raw[idx: idx + nWerke]]
    stages = {w: int(s) for w, s in Werke_raw}
    idx += nWerke

    # next line is number of streets - tolerate labels
    nStrassen_line = raw[idx]
    try:
        nStrassen = int(nStrassen_line)
    except ValueError:
        parts = nStrassen_line.split()
        nStrassen = int(parts[-1])
    idx += 1

    street_lines = [ln.split() for ln in raw[idx: idx + nStrassen]]

    Strassen = []
    S_node = None
    T_node = None
    for parts in street_lines:
        if len(parts) < 3:
            continue
        u, v, w = parts[0], parts[1], parts[2]
        try:
            wi = int(w)
        except ValueError:
            # fallback: last token
            wi = int(parts[-1])
        Strassen.append((u, v, wi))
        # remember S/T
        if u == 'S' or v == 'S':
            S_node = 'S'
        if u == 'T' or v == 'T':
            T_node = 'T'

    # make bidirectional (add reversed edges)
    Strassen_bidirectional = []
    for u, v, w in Strassen:
        Strassen_bidirectional.append((u, v, w))
        Strassen_bidirectional.append((v, u, w))

    return T, stages, Strassen_bidirectional, S_node, T_node


def create_adj_dict(Strassen):
    """Return adjacency dict {node: [[neighbor, weight], ...]} ensuring all nodes appear.
    The graph is treated as undirected: for every (u,v,w) we add both u->v and v->u.
    Duplicate parallel entries (same neighbor and weight) are avoided.
    """
    adj = {}
    for u, v, w in Strassen:
        wi = int(w)
        # ensure lists exist
        adj.setdefault(u, [])
        adj.setdefault(v, [])
        # add u -> v if not present
        if not any(ne == v and wt == wi for ne, wt in adj[u]):
            adj[u].append([v, wi])
        # add v -> u if not present (undirected)
        if not any(ne == u and wt == wi for ne, wt in adj[v]):
            adj[v].append([u, wi])
    return adj


def dijkstra_distance(adj, source, target):
    """Return distance (int or inf) from source to target using Dijkstra.
    Works even if source/target not in adj keys by adding them to the node set.
    """
    nodes = set(adj.keys()) | {source, target}
    dist = {n: float('inf') for n in nodes}
    dist[source] = 0
    unvisited = set(nodes)

    while unvisited:
        u = min(unvisited, key=lambda x: dist[x])
        if dist[u] == float('inf'):
            break
        unvisited.remove(u)

        if u == target:
            return dist[u]

        for v, w in adj.get(u, []):
            if v in unvisited:
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd

    return dist.get(target, float('inf'))


def dijkstra_path(adj, source, target):
    """Return shortest path as list of nodes from source to target (or [] if none)."""
    nodes = set(adj.keys()) | {source, target}
    dist = {n: float('inf') for n in nodes}
    prev = {n: None for n in nodes}
    dist[source] = 0
    unvisited = set(nodes)

    while unvisited:
        u = min(unvisited, key=lambda x: dist[x])
        if dist[u] == float('inf'):
            break
        unvisited.remove(u)

        for v, w in adj.get(u, []):
            if v in unvisited:
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u

    if dist.get(target, float('inf')) == float('inf'):
        return []

    path = []
    cur = target
    while cur is not None:
        path.insert(0, cur)
        cur = prev[cur]
    return path


def restrict_to_same_stage(adj, stages, strike_node):
    """Restrict outgoing edges from strike_node to nodes in the same stage only.
    If strike_node not in stages, return adj unchanged (no restriction).
    """
    if strike_node not in stages:
        return {u: [e.copy() for e in edges] for u, edges in adj.items()}

    strike_stage = stages[strike_node]
    new_adj = {}

    for u, edges in adj.items():
        if u != strike_node:
            new_adj[u] = [e.copy() for e in edges]
        else:
            new_adj[u] = [
                [v, w] for v, w in edges
                if stages.get(v) == strike_stage
            ]
    return new_adj


def simulate_strike_times(adj, stages, source="S", target="T"):
    base_path = dijkstra_path(adj, source, target)
    if not base_path:
        return {}

    results = {}

    for strike_node in base_path[:-1]:
        # skip nodes that are not works (not in stages)
        if strike_node not in stages:
            continue
        modified = restrict_to_same_stage(adj, stages, strike_node)
        time = dijkstra_distance(modified, strike_node, target)
        results[strike_node] = time

    return results


def main():
    # make base_dir relative to this script file so the runner works from repo root
    base_dir = os.path.join(os.path.dirname(__file__), "A3_Lieferkette")

    for fname in sorted(os.listdir(base_dir)):
        if not fname.startswith("lieferung"):
            continue

        path = os.path.join(base_dir, fname)
        T_max, stages, Strassen, S_node, T_node = read_file(path)
        adj = create_adj_dict(Strassen)

        strike_times = simulate_strike_times(
            adj,
            stages,
            source="S",
            target="T"
        )

        print(f"\n===== {fname} =====")
        for node, time in strike_times.items():
            if time == float('inf'):
                print(f"Strike at {node}: ❌ NO PATH")
            elif time <= T_max:
                print(f"Strike at {node}: ✅ {time} ≤ {T_max}")
            else:
                print(f"Strike at {node}: ❌ {time} > {T_max}")


if __name__ == "__main__":
    main()
