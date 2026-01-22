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

    # make bidirectional and ensure weights are ints
    Strassen_bidirectional = []
    for u, v, w in Strassen:
        Strassen_bidirectional.append([u, v, int(w)])
    for u, v, w in Strassen_bidirectional.copy():
        Strassen_bidirectional.append([v, u, int(w)])

    return T, Werke, Strassen_bidirectional


def create_adj_dict(Strassen, Werke=None):
    adj = {}
    for u, v, w in Strassen:
        adj.setdefault(u, []).append([v, int(w)])
    # ensure nodes from Werke exist in adj even if they have no outgoing edges
    if Werke:
        for n in Werke.keys():
            adj.setdefault(n, [])
    return adj


def dijkstra_all(adj, source):
    # Dijkstra that returns distances and predecessors (heap-based implementation)
    import heapq
    INF = float('inf')
    dist = {n: INF for n in adj.keys()}
    prev = {n: None for n in adj.keys()}
    if source not in dist:
        return dist, prev
    dist[source] = 0
    heap = [(0, source)]

    while heap:
        d_u, u = heapq.heappop(heap)
        # stale entry check
        if d_u != dist[u]:
            continue
        for v, w in adj.get(u, []):
            nd = d_u + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return dist, prev


def dijkstra_path_from_prev(prev, source, target):
    if target not in prev:
        return []
    if prev[target] is None and source != target:
        # maybe direct unreachable
        # attempt to build path if target==source
        if source == target:
            return [source]
        return []
    path = []
    cur = target
    while cur is not None:
        path.insert(0, cur)
        if cur == source:
            break
        cur = prev.get(cur)
    return path


def dijkstra_distance(adj, source, target):
    dist, _ = dijkstra_all(adj, source)
    return dist.get(target, float('inf'))


def dijkstra_path(adj, source, target):
    dist, prev = dijkstra_all(adj, source)
    if dist.get(target, float('inf')) == float('inf'):
        return []
    return dijkstra_path_from_prev(prev, source, target)


def remove_node(adj, node):
    new_adj = {}
    for u, edges in adj.items():
        if u == node:
            continue
        new_edges = [[v, w] for v, w in edges if v != node]
        new_adj[u] = new_edges
    return new_adj


def _compute_stage_nodes(stages):
    m = 0
    stage_nodes = {}
    for n, s in stages.items():
        stage_nodes.setdefault(s, []).append(n)
        if s > m:
            m = s
    return m, stage_nodes


def _precompute_distances(adj, sources):
    # Run Dijkstra from each source in 'sources' and return dict of dicts
    dist_map = {}
    for s in sources:
        dist_map[s], _ = dijkstra_all(adj, s)
    return dist_map


def find_best_sequence(adj, stages, source='S', target='T'):
    """Find best sequence (one plant per stage in order) minimizing total travel time.
    Returns (total_cost, sequencelist) where sequencelist is [a1,...,am] or (inf, [])."""
    m, stage_nodes = _compute_stage_nodes(stages)
    # validate stages exist
    for k in range(1, m + 1):
        if k not in stage_nodes or not stage_nodes[k]:
            return float('inf'), []

    # precompute distances from S and from every plant node
    sources = {source, target} | set(stages.keys())
    dist_map = _precompute_distances(adj, sources)

    INF = float('inf')
    # DP over stages
    dp_prev = {}
    prev_choice = {}  # (k,node) -> previous node
    # stage 1
    for p in stage_nodes[1]:
        dp_prev[p] = dist_map[source].get(p, INF)
        prev_choice[(1, p)] = source

    for k in range(2, m + 1):
        dp_cur = {}
        for p2 in stage_nodes[k]:
            best = INF
            best_prev = None
            for p1, cost1 in dp_prev.items():
                d = dist_map[p1].get(p2, INF)
                tot = cost1 + d
                if tot < best:
                    best = tot
                    best_prev = p1
            if best < INF:
                dp_cur[p2] = best
                prev_choice[(k, p2)] = best_prev
        dp_prev = dp_cur
        if not dp_prev:
            return INF, []

    # choose final plant minimizing dp_prev[p] + dist to T
    best_total = INF
    best_last = None
    for p, cost in dp_prev.items():
        dT = dist_map[p].get(target, INF)
        tot = cost + dT
        if tot < best_total:
            best_total = tot
            best_last = p

    if best_last is None:
        return INF, []

    # reconstruct sequence
    seq = [None] * m
    cur = best_last
    for k in range(m, 0, -1):
        seq[k - 1] = cur
        cur = prev_choice.get((k, cur))
    return best_total, seq


def _compute_tail_from_node(adj, stages, start_node, start_stage_index, dist_map):
    """Given that start_node is chosen for stage start_stage_index, compute minimal tail sequence and cost from that node through stages start_stage_index..m and to T.
    Returns (tail_cost_to_T, tail_sequence_for_stages_starting_at_index) where sequence includes start_node as stage start_stage_index."""
    m, stage_nodes = _compute_stage_nodes(stages)
    INF = float('inf')
    # DP starting with start_node as stage start_stage_index
    dp_prev = {start_node: 0}
    prev_choice = {}

    for k in range(start_stage_index + 1, m + 1):
        dp_cur = {}
        for p2 in stage_nodes[k]:
            best = INF
            best_prev = None
            for p1, cost1 in dp_prev.items():
                d = dist_map[p1].get(p2, INF)
                tot = cost1 + d
                if tot < best:
                    best = tot
                    best_prev = p1
            if best < INF:
                dp_cur[p2] = best
                prev_choice[(k, p2)] = best_prev
        dp_prev = dp_cur
        if not dp_prev:
            return INF, []

    # choose last plant and reconstruct
    best_total = INF
    best_last = None
    for p, cost in dp_prev.items():
        dT = dist_map[p].get('T', INF)
        tot = cost + dT
        if tot < best_total:
            best_total = tot
            best_last = p

    if best_last is None:
        return INF, []

    # reconstruct sequence including start_node at position start_stage_index
    seq = [None] * (m - start_stage_index + 1)
    seq[-1] = best_last
    cur = best_last
    for k in range(m, start_stage_index, -1):
        prevn = prev_choice.get((k, cur))
        seq[k - start_stage_index - 1] = prevn
        cur = prevn
    seq[0] = start_node
    return best_total, seq


def simulate_strike_times(adj, stages, source="S", target="T"):
    """Return a dict mapping each node on the chosen base sequence to a dict with keys:
       'base_total' : total time of the base plan,
       'base_sequence': [a1..am],
       'strike_alternative': (best_time_if_strike, alternative_sequence_including_replacement)
    """
    # compute base plan
    base_total, base_seq = find_best_sequence(adj, stages, source, target)
    if base_total == float('inf'):
        return {'__base__': {'base_total': float('inf'), 'base_sequence': []}}

    # precompute distances from S and all plant nodes
    sources = {source, target} | set(stages.keys())
    dist_map = _precompute_distances(adj, sources)

    # compute prefix costs to reach each base_seq[i]
    prefix_costs = []
    cur_cost = 0
    if base_seq:
        # cost to reach first plant
        cur_cost = dist_map[source].get(base_seq[0], float('inf'))
        prefix_costs.append(cur_cost)
        for i in range(1, len(base_seq)):
            d = dist_map[base_seq[i - 1]].get(base_seq[i], float('inf'))
            cur_cost = cur_cost + d
            prefix_costs.append(cur_cost)

    results = {'__base__': {'base_total': base_total, 'base_sequence': base_seq}}

    m, stage_nodes = _compute_stage_nodes(stages)

    for i, struck in enumerate(base_seq):
        stage_index = stages[struck]
        # candidates are other plants in the same stage
        candidates = [v for v in stage_nodes.get(stage_index, []) if v != struck]
        best_alt = float('inf')
        best_seq = []
        if not candidates:
            results[struck] = {'best_time': float('inf'), 'alt_sequence': []}
            continue

        # prefix cost to arrive at struck plant
        time_to_struck = prefix_costs[i]

        # for each candidate replacement, compute tail cost starting from candidate
        for b in candidates:
            # driver must drive from struck location to b
            d_struck_to_b = dist_map[struck].get(b, float('inf'))
            if d_struck_to_b == float('inf'):
                continue
            tail_cost, tail_seq = _compute_tail_from_node(adj, stages, b, stage_index, dist_map)
            if tail_cost == float('inf'):
                continue
            total = time_to_struck + d_struck_to_b + tail_cost
            if total < best_alt:
                best_alt = total
                # build full sequence: base_seq up to i-1, replacement b, then tail_seq[1:] (exclude b duplicate)
                full_seq = base_seq[:i] + tail_seq
                best_seq = full_seq

        results[struck] = {'best_time': best_alt, 'alt_sequence': best_seq}

    return results


def main():
    # make base_dir relative to this script location so script runs from any CWD
    base_dir = os.path.join(os.path.dirname(__file__), "A3_Lieferkette")

    for fname in sorted(os.listdir(base_dir)):
        if not fname.startswith("lieferung"):
            continue

        path = os.path.join(base_dir, fname)
        T_max, stages, Strassen = read_file(path)
        adj = create_adj_dict(Strassen, Werke=stages)

        # compute and print the normal Dijkstra path (S -> ... -> T)
        base_path = dijkstra_path(adj, "S", "T")
        base_path_str = " -> ".join(base_path) if base_path else "NO PATH"

        strike_times = simulate_strike_times(
            adj,
            stages,
            source="S",
            target="T"
        )

        print(f"\n===== {fname} =====")
        # print base plan info from the simulator
        base_info = strike_times.get('__base__', {})
        base_total = base_info.get('base_total', float('inf'))
        base_seq = base_info.get('base_sequence', [])
        base_seq_str = ' -> '.join(base_seq) if base_seq else 'NO PLAN'
        if base_total == float('inf'):
            print(f"Base plan: NO PLAN")
        else:
            print(f"Base total time: {base_total}")
            print(f"Base sequence: {base_seq_str}")

        # per-strike alternatives (iterate in order of base sequence)
        for node in base_seq:
            info = strike_times.get(node, {})
            best_time = info.get('best_time', float('inf'))
            alt_seq = info.get('alt_sequence', [])
            alt_seq_str = ' -> '.join(alt_seq) if alt_seq else 'NO PATH'
            if best_time == float('inf'):
                print(f"Strike at {node}: NO PATH")
            elif best_time <= T_max:
                print(f"Strike at {node}: {best_time} ≤ {T_max}; alt: {alt_seq_str}")
            else:
                print(f"Strike at {node}: {best_time} > {T_max}; alt: {alt_seq_str}")


if __name__ == "__main__":
    main()
