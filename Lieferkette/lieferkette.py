import networkx as nx
import matplotlib.pyplot as plt


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


def dijkstra_shortest_path(Streets, source, target):
    nodes = set()
    for start, end, weight in Streets:
        nodes.add(start)
        nodes.add(end)

    unvisited = nodes
    distances = {source: 0}
    previous = {}

    while unvisited:
        current_node = min(unvisited, key=lambda node: distances.get(node, float('inf')))

        if distances.get(current_node, float('inf')) == float('inf') or current_node == target:
            break

        unvisited.remove(current_node)

        for start, end, weight in Streets:
            if start == current_node:
                neighbor = end
                weight_val = int(weight)
                new_distance = distances[current_node] + weight_val

                if new_distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node

    path = []
    curr = target
    while curr in previous:
        path.insert(0, curr)
        curr = previous[curr]
    if path:
        path.insert(0, source)

    return path, distances.get(target, float("inf"))


def create_layered_graph(Werke, Strassen):
    factory_to_stage = {factory_id: int(stage) for factory_id, stage in Werke.items()}

    factories_by_stage = {}
    for factory_id, stage in factory_to_stage.items():
        factories_by_stage.setdefault(stage, []).append(factory_id)

    layered_edges = []
    stages = sorted(factories_by_stage.keys())

    for factory in factories_by_stage[stages[0]]:
        _, distance = dijkstra_shortest_path(Strassen, "S", factory)
        if distance != float('inf'):
            layered_edges.append(["S", factory, int(distance)])

    for i in range(len(stages) - 1):
        for u in factories_by_stage[stages[i]]:
            for v in factories_by_stage[stages[i + 1]]:
                _, distance = dijkstra_shortest_path(Strassen, u, v)
                if distance != float('inf'):
                    layered_edges.append([u, v, int(distance)])

    for factory in factories_by_stage[stages[-1]]:
        _, distance = dijkstra_shortest_path(Strassen, factory, "T")
        if distance != float('inf'):
            layered_edges.append([factory, "T", int(distance)])

    return layered_edges


T, Schritte, WerkeSchritte, nWerke, Werke, nStrassen, Strassen = read_file("A3_Lieferkette/lieferung00.txt")

path, distance = dijkstra_shortest_path(Strassen, "S", "T")

layered_graph = create_layered_graph(Werke, Strassen)

path_layered, distance = dijkstra_shortest_path(layered_graph, "S", "T")
print(f"Bester Pfad durch die Werke (Schichtengraph): {path_layered}")
print(f"Gesamtzeit: {distance}")

full_path = []
if path_layered:
    for i in range(len(path_layered) - 1):
        segment_path, _ = dijkstra_shortest_path(Strassen, path_layered[i], path_layered[i + 1])
        if i == 0:
            full_path.extend(segment_path)
        else:
            full_path.extend(segment_path[1:])

print(f"Vollstandiger Pfad im Strassennetz: {full_path}")

if int(distance) <= int(T):
    print("Die Lieferung ist innerhalb der Zeitfrist moglich.")
else:
    print("Die Lieferung ist NICHT innerhalb der Zeitfrist moglich.")

visualize_graph(Strassen, full_path)

print("\nStreik Scenarios testen")
print(f"Optimaler Pfad ohne Streik: {path_layered}")
print(f"Optimale Zeit ohne Streik: {distance}")
print()

factories_to_test = [f for f in path_layered if f != "S" and f != "T"]
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
    strike_stage = factory_to_stage[strike_factory]

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
        path, distance_alt = dijkstra_shortest_path(Strassen, strike_factory, alt_factory)
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
            segment_path, _ = dijkstra_shortest_path(Strassen, path_with_strike[i], path_with_strike[i+1])
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


