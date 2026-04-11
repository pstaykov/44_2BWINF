import os

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f.readlines()]

    T = int(raw[0])
    Schritte = raw[1]
    WerkeSchritte = raw[2].split()
    nWerke = int(raw[3])

    Werke_raw = [ln.split() for ln in raw[4:4 + nWerke]]
    Werke = {w: int(s) for w, s in Werke_raw}

    nStrassen = int(raw[4 + nWerke])
    Strassen = [ln.split() for ln in raw[5 + nWerke:]]

    # make bidirectional
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
    if Werke:
        for n in Werke.keys():
            adj.setdefault(n, [])
    return adj


def dijkstra_all(adj, source):
    """
    :param adj: adjacency list as dict: node -> list of (neighbor, weight)
    :param source: starting node
    :return: best distance dict from source to all nodes, best previous dict from source to all nodes
    """
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
        if d_u != dist[u]:
            continue
        for v, w in adj.get(u, []):
            nd = d_u + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return dist, prev

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
    """
    :param adj: adjacency list as dict: node -> list of (neighbor, weight)
    :param stages: node -> stage
    :param source: starting node
    :param target: ending node
    :return: (total_time, sequence)
    """
    m, stage_nodes = _compute_stage_nodes(stages)
    for k in range(1, m + 1):
        if k not in stage_nodes or not stage_nodes[k]:
            return float('inf'), []

    # precompute distances from S and from every plant node
    sources = {source, target} | set(stages.keys())
    dist_map = _precompute_distances(adj, sources)

    INF = float('inf')
    dp_prev = {}
    prev_choice = {}
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
    """
    :param adj: adjacency list as dict: node -> list
    :param stages: node -> stage
    :param start_node: starting node
    :param start_stage_index: path up to start_node is stage start_stage_index
    :param dist_map: map from node to dict of distances to all nodes
    :return: best_total_time, sequence
    """
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
    """
    :param adj: adjacency list as dict: node -> list
    :param stages: node -> stage
    :param source: starting node
    :param target: target node
    :return: dict with keys:
    - '__base__': {'base_total': total_time, 'base_sequence': sequence}
    - for each plant in base_sequence: {plant_node: {'best_time': best_time, 'alt_sequence': alt_sequence}}
     where best_time is the best total time if the plant is struck and alt_sequence is the best sequence of plants to use starting with the struck plant and then the tail plants ending with T.
    """
    base_total, base_seq = find_best_sequence(adj, stages, source, target)
    if base_total == float('inf'):
        return {'__base__': {'base_total': float('inf'), 'base_sequence': []}}

    sources = {source, target} | set(stages.keys())
    dist_map = _precompute_distances(adj, sources)

    prefix_costs = []
    cur_cost = 0
    if base_seq:
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
        candidates = [v for v in stage_nodes.get(stage_index, []) if v != struck]
        best_alt = float('inf')
        best_seq = []
        if not candidates:
            results[struck] = {'best_time': float('inf'), 'alt_sequence': []}
            continue
        time_to_struck = prefix_costs[i]

        for b in candidates:
            d_struck_to_b = dist_map[struck].get(b, float('inf'))
            if d_struck_to_b == float('inf'):
                continue
            tail_cost, tail_seq = _compute_tail_from_node(adj, stages, b, stage_index, dist_map)
            if tail_cost == float('inf'):
                continue
            total = time_to_struck + d_struck_to_b + tail_cost
            if total < best_alt:
                best_alt = total
                full_seq = [struck] + tail_seq
                best_seq = full_seq

        results[struck] = {'best_time': best_alt, 'alt_sequence': best_seq}

    return results


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "A3_Lieferkette")
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(base_dir)):
        if not fname.startswith("lieferung"):
            continue

        path = os.path.join(base_dir, fname)
        T_max, stages, Strassen = read_file(path)
        adj = create_adj_dict(Strassen, Werke=stages)

        strike_times = simulate_strike_times(
            adj,
            stages,
            source="S",
            target="T"
        )

        base_info = strike_times.get('__base__', {})
        base_total = base_info.get('base_total', float('inf'))
        base_seq = base_info.get('base_sequence', [])

        out_lines = []
        if base_total == float('inf'):
            # unsolvable
            out_lines.append("UNMOEGLICH")
            out_path = os.path.join(out_dir, fname + "_out.txt")
            with open(out_path, 'w', encoding='utf-8') as fo:
                fo.write('\n'.join(out_lines) + '\n')
            continue

        # solvable
        out_lines.append("MOEGLICH")

        # compute maximal route duration that can occur after the plan
        max_dur = base_total
        any_inf = False
        for node in base_seq:
            info = strike_times.get(node, {})
            best_time = info.get('best_time', float('inf'))
            if best_time == float('inf'):
                any_inf = True
            else:
                if best_time > max_dur:
                    max_dur = best_time

        if any_inf:
            out_lines.append("UNMOEGLICH")
        else:
            out_lines.append(str(int(max_dur)))

        out_lines.append("")

        out_lines.append(str(int(base_total)))
        if base_seq:
            base_route = 'S ' + ' '.join(f'[{n} {stages[n]}]' for n in base_seq) + ' T'
            out_lines.append(base_route)
        else:
            out_lines.append('S T')

        for node in base_seq:
            info = strike_times.get(node, {})
            best_time = info.get('best_time', float('inf'))
            alt_seq = info.get('alt_sequence', [])
            if best_time == float('inf') or not alt_seq:
                out_lines.append("UNMOEGLICH")
            else:
                out_lines.append(str(int(best_time)))
                if len(alt_seq) <= 1:
                    out_lines.append(f"{alt_seq[0]} T")
                else:
                    rest = ' '.join(f'[{n} {stages[n]}]' for n in alt_seq[1:])
                    out_lines.append(f"{alt_seq[0]} {rest} T")

        # final blank line
        out_lines.append("")

        out_path = os.path.join(out_dir, fname + "_out.txt")
        with open(out_path, 'w', encoding='utf-8') as fo:
            fo.write('\n'.join(out_lines) + '\n')


if __name__ == "__main__":
    main()
