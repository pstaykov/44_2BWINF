import os

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f.readlines()]

    T = int(raw[0])
    Schritte = raw[1]
    WerkeSchritte = raw[2].split()  # list like ["A", "1", "B", "2", ...]
    nWerke = int(raw[3])

    Werke_raw = [ln.split() for ln in raw[4:4 + nWerke]]
    Werke = {w: int(s) for w, s in Werke_raw}

    nStrassen = int(raw[4 + nWerke])
    Strassen = [ln.split() for ln in raw[5 + nWerke:]]

    # make bidirectional
    Strassen_bidirectional = Strassen.copy()
    for u, v, w in Strassen:
        Strassen_bidirectional.append([v, u, w])

    return T, Werke, Strassen_bidirectional

def create_adj_dict(Strassen):
    adj = {}
    for u, v, w in Strassen:
        adj.setdefault(u, []).append([v, int(w)])
    return adj

def dijkstra_distance(adj, source, target):
    unvisited = set(adj.keys())
    dist = {n: float('inf') for n in unvisited}
    dist[source] = 0

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
    unvisited = set(adj.keys())
    dist = {n: float('inf') for n in unvisited}
    prev = {n: None for n in unvisited}
    dist[source] = 0

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

    if dist[target] == float('inf'):
        return []

    path = []
    cur = target
    while cur:
        path.insert(0, cur)
        cur = prev[cur]
    return path

def restrict_to_same_stage(adj, stages, strike_node):
    strike_stage = stages[strike_node]
    new_adj = {}

    for u, edges in adj.items():
        if u != strike_node:
            new_adj[u] = edges.copy()
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
        modified = restrict_to_same_stage(adj, stages, strike_node)
        time = dijkstra_distance(modified, strike_node, target)
        results[strike_node] = time

    return results


def main():
    base_dir = "A3_Lieferkette"

    for fname in sorted(os.listdir(base_dir)):
        if not fname.startswith("lieferung"):
            continue

        path = os.path.join(base_dir, fname)
        T_max, stages, Strassen = read_file(path)
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
