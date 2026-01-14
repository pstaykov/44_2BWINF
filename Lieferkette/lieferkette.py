import networkx as nx
import matplotlib.pyplot as plt
import heapq


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]
    T = raw_lines[0]
    Schritte = raw_lines[1]
    WerkeSchritte = raw_lines[2].split()
    nWerke = int(raw_lines[3])
    Werke = [ln.split() for ln in raw_lines[4:4 + nWerke]]
    Werke = {ln[0]: ln[1] for ln in Werke}
    nStrassen = raw_lines[nWerke + 4]
    Strassen = [l.split() for l in raw_lines[nWerke + 5:]]

    # Make the graph bidirectional by adding reverse edges
    # Add reverse edges for all streets (guaranteed no duplicate edges)
    Strassen_bidirectional = Strassen.copy()
    for street in Strassen:
        start, end, weight = street[0], street[1], street[2]
        Strassen_bidirectional.append([end, start, weight])

    return T, Schritte, WerkeSchritte, nWerke, Werke, nStrassen, Strassen_bidirectional


def visualize_graph(Strassen, path=None):
    G = nx.DiGraph()
    for start, end, time in Strassen:
        G.add_edge(start, end, weight=int(time))

    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, arrows=True)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange', node_size=2000)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()


def build_adjacency_list(edge_list):
    """Convert edge list [[u,v,w], ...] into adjacency dict {u: [(v,w), ...], ...}."""
    adj = {}
    for u, v, w in edge_list:
        try:
            wt = int(w)
        except Exception:
            wt = float('inf')
        adj.setdefault(u, []).append((v, wt))
        # ensure nodes without outgoing edges appear in adj
        adj.setdefault(v, adj.get(v, []))
    return adj


def reverse_graph(adj):
    rev = {}
    for u, nbrs in adj.items():
        rev.setdefault(u, [])
        for v, w in nbrs:
            rev.setdefault(v, []).append((u, w))
    return rev


def dijkstra_single_source(adj, source, targets=None, return_prev=False):
    """Heap-based Dijkstra from source over adjacency dict.

    Args:
        adj: dict {node: [(nbr, weight), ...]}
        source: start node
        targets: optional set of target nodes to early-stop when all found
        return_prev: whether to also return predecessor map

    Returns:
        distances dict, and optionally prev dict
    """

    distances = {source: 0}
    prev = {}
    seen = set()
    heap = [(0, source)]
    targets = set(targets) if targets is not None else None
    found_targets = set()

    while heap:
        dist_u, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)

        if targets is not None and u in targets:
            found_targets.add(u)
            if len(found_targets) == len(targets):
                break

        for v, w in adj.get(u, []):
            newd = dist_u + w
            if newd < distances.get(v, float('inf')):
                distances[v] = newd
                prev[v] = u
                heapq.heappush(heap, (newd, v))

    if return_prev:
        return distances, prev
    return distances


def dijkstra_shortest_path(Streets, source, target):
    """
    Backwards-compatible wrapper that accepts either an adjacency dict or an edge list.
    Returns (path_list, distance).
    """
    # Build adjacency list only if Streets is not already an adjacency dict
    if isinstance(Streets, dict):
        adj = Streets
    else:
        adj = build_adjacency_list(Streets)

    # Run single-source Dijkstra with early stop on target
    distances, prev = dijkstra_single_source(adj, source, targets={target}, return_prev=True)
    dist = distances.get(target, float('inf'))

    path = []
    if target in prev or (target == source and target in distances):
        # reconstruct path
        cur = target
        path.append(cur)
        while cur != source:
            cur = prev.get(cur)
            if cur is None:
                break
            path.insert(0, cur)
    else:
        # No path found
        path = []

    return path, dist


def create_layered_graph(Werke, Strassen):
    # Build adjacency once and use Dijkstra from each source to the whole next stage
    factory_to_stage = {factory_id: int(stage) for factory_id, stage in Werke.items()}

    factories_by_stage = {}
    for factory_id, stage in factory_to_stage.items():
        factories_by_stage.setdefault(stage, []).append(factory_id)

    layered_edges = []
    stages = sorted(factories_by_stage.keys())

    # build adjacency and reversed adjacency
    adj = build_adjacency_list(Strassen)
    rev_adj = reverse_graph(adj)

    # S -> first stage: one run from S to all factories in first stage
    first_stage_factories = factories_by_stage.get(stages[0], [])
    if first_stage_factories:
        distances = dijkstra_single_source(adj, "S", targets=set(first_stage_factories))
        for factory in first_stage_factories:
            d = distances.get(factory, float('inf'))
            if d != float('inf'):
                layered_edges.append(["S", factory, int(d)])

    # between stages: for each factory in current stage, run Dijkstra once to all factories in next stage
    for i in range(len(stages) - 1):
        current_stage = factories_by_stage.get(stages[i], [])
        next_stage = factories_by_stage.get(stages[i + 1], [])
        next_set = set(next_stage)
        for u in current_stage:
            distances = dijkstra_single_source(adj, u, targets=next_set)
            for v in next_stage:
                d = distances.get(v, float('inf'))
                if d != float('inf'):
                    layered_edges.append([u, v, int(d)])

    # last stage -> T: run Dijkstra on reversed graph from T to all last-stage factories
    last_stage_factories = factories_by_stage.get(stages[-1], [])
    if last_stage_factories:
        distances_to_last = dijkstra_single_source(rev_adj, "T", targets=set(last_stage_factories))
        for factory in last_stage_factories:
            d = distances_to_last.get(factory, float('inf'))
            if d != float('inf'):
                layered_edges.append([factory, "T", int(d)])

    return layered_edges



def process_file(file_path, visualize=False):
    try:
        T, Schritte, WerkeSchritte, nWerke, Werke, nStrassen, Strassen = read_file(file_path)
    except FileNotFoundError:
        print(f"Datei nicht gefunden: {file_path}")
        return None

    print("\n" + "=" * 60)
    print(f"Verarbeite: {file_path}")

    # Build adjacency once for the street network so repeated shortest-paths are fast
    adj = build_adjacency_list(Strassen)

    path, distance = dijkstra_shortest_path(adj, "S", "T")

    layered_graph = create_layered_graph(Werke, Strassen)

    path_layered, distance = dijkstra_shortest_path(layered_graph, "S", "T")
    print(f"Bester Pfad durch die Werke (Schichtengraph): {path_layered}")
    print(f"Gesamtzeit: {distance}")

    full_path = []
    if path_layered:
        for i in range(len(path_layered) - 1):
            segment_path, _ = dijkstra_shortest_path(adj, path_layered[i], path_layered[i + 1])
            if i == 0:
                full_path.extend(segment_path)
            else:
                full_path.extend(segment_path[1:])

    print(f"Vollstandiger Pfad im Strassennetz: {full_path}")

    try:
        if int(distance) <= int(T):
            print("Die Lieferung ist innerhalb der Zeitfrist moglich.")
        else:
            print("Die Lieferung ist NICHT innerhalb der Zeitfrist moglich.")
    except (ValueError, TypeError):
        print("Warnung: Ungultige Zeitangabe T oder distance; Ueberpruefen der Eingabedatei.")

    if visualize:
        visualize_graph(Strassen, full_path)

    print("\nStreik Scenarios testen")
    print(f"Optimaler Pfad ohne Streik: {path_layered}")
    print(f"Optimale Zeit ohne Streik: {distance}")
    print()

    factories_to_test = [f for f in path_layered if f != "S" and f != "T"] if path_layered else []
    all_feasible = True
    contingency_plans = {}

    for strike_factory in factories_to_test:
        print(f"Szenario: Werk {strike_factory} ist bestreikt")

        strike_index = path_layered.index(strike_factory)
        path_up_to_strike = path_layered[:strike_index + 1]

        time_up_to_strike = 0
        for i in range(len(path_up_to_strike) - 1):
            for edge in layered_graph:
                if edge[0] == path_up_to_strike[i] and edge[1] == path_up_to_strike[i + 1]:
                    time_up_to_strike += edge[2]
                    break

        factory_to_stage = {factory_id: int(stage) for factory_id, stage in Werke.items()}
        strike_stage = factory_to_stage.get(strike_factory)

        factories_in_same_stage = [fid for fid, stage in factory_to_stage.items() if stage == strike_stage and fid != strike_factory]

        filtered_layered_graph = []
        for edge in layered_graph:
            source, target, weight = edge[0], edge[1], edge[2]

            if target == strike_factory:
                continue

            if source == strike_factory:
                if target in factories_in_same_stage:
                    filtered_layered_graph.append(edge)
            else:
                filtered_layered_graph.append(edge)

        for alt_factory in factories_in_same_stage:
            path, distance_alt = dijkstra_shortest_path(adj, strike_factory, alt_factory)
            if distance_alt != float('inf'):
                filtered_layered_graph.append([strike_factory, alt_factory, int(distance_alt)])

        path_from_strike, distance_from_strike = dijkstra_shortest_path(filtered_layered_graph, strike_factory, "T")

        if path_from_strike and distance_from_strike != float('inf'):
            path_with_strike = path_up_to_strike + path_from_strike[1:]
            distance_with_strike = time_up_to_strike + distance_from_strike
        else:
            path_with_strike = []
            distance_with_strike = float('inf')

        print(f"Notfall-Pfad: {path_with_strike}")
        print(f"Zeit mit Streik: {distance_with_strike}")

        if distance_with_strike != float('inf') and int(distance_with_strike) <= int(T):
            print(f"Lieferung moglich innerhalb Zeitfrist ({distance_with_strike} <= {T})")
        else:
            if distance_with_strike == float('inf'):
                print(f"Lieferung nicht moglich (kein alternativer Weg gefunden)")
            else:
                print(f"Lieferung nicht moglich ({distance_with_strike} > {T})")
            all_feasible = False

        full_path_strike = []
        if path_with_strike and distance_with_strike != float('inf'):
            for i in range(len(path_with_strike) - 1):
                segment_path, _ = dijkstra_shortest_path(adj, path_with_strike[i], path_with_strike[i+1])
                if i == 0:
                    full_path_strike.extend(segment_path)
                else:
                    full_path_strike.extend(segment_path[1:])

        contingency_plans[strike_factory] = {
            'path_layered': path_with_strike,
            'path_full': full_path_strike,
            'distance': distance_with_strike,
            'feasible': distance_with_strike != float('inf') and int(distance_with_strike) <= int(T)
        }

        print(f"Vollstandiger Pfad: {full_path_strike}")
        print()

    print("\nZUSAMMENFASSUNG: STREIK-AUSWIRKUNGEN")
    print("=" * 50)

    possible_strikes = [factory for factory, plan in contingency_plans.items() if plan['feasible']]
    impossible_strikes = [factory for factory, plan in contingency_plans.items() if not plan['feasible']]

    if possible_strikes:
        print(f"Lieferung MOGLICH bei Streik von: {', '.join(possible_strikes)}")
    else:
        print("Keine Werke, bei deren Streik die Lieferung moglich ist.")

    if impossible_strikes:
        print(f"Lieferung NICHT MOGLICH bei Streik von: {', '.join(impossible_strikes)}")
    else:
        print("Alle Werke: Die Lieferung bleibt bei Streik moglich.")

    print("=" * 50)

    return {
        'file': file_path,
        'path_layered': path_layered,
        'distance': distance,
        'full_path': full_path,
        'contingency': contingency_plans,
        'all_feasible': all_feasible
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prozessiere Lieferungsdateien (lieferung00..lieferung08).')
    parser.add_argument('--start', type=int, default=0, help='Startindex (inclusive), default 0')
    parser.add_argument('--end', type=int, default=8, help='Endindex (inclusive), default 8')
    parser.add_argument('--visualize', action='store_true', help='Zeige Graphen fuer jede Datei')
    args = parser.parse_args()

    results = []
    for i in range(args.start, args.end + 1):
        fname = f"A3_Lieferkette/lieferung{i:02d}.txt"
        res = process_file(fname, visualize=args.visualize)
        if res is not None:
            results.append(res)

    print("\nVerarbeitung abgeschlossen. Zusammenfassung:\n")
    for r in results:
        print(f"{r['file']}: distance={r['distance']}, feasible_all_strikes={r['all_feasible']}")
