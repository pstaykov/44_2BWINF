import networkx as nx
import matplotlib.pyplot as plt
import heapq


def read_file(path):
    '''
    Strassen variabel:
    Ausgangspunkt der Straße;
    Endpunkt der Straße;
    zum Befahren der Straße benötigte Zeit.
    '''
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
    # Add reverse edges for all streets
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

def create_adj_dict(Strassen):
    """
    converts input to adj list of the following form:
    {node: [[neighbor, distance]}
    """
    adj_list = {}
    for street in Strassen:
        if street[0] not in adj_list:
            adj_list[street[0]] = [[street[1], street[2]]]
        else:
            adj_list[street[0]].append([street[1], street[2]])

    return adj_list

def dijkstra(Strassen, source, targets):
    """
    input: strassen as a adj dictionary
    output: distance to all targets
    """
    unvisited = Strassen.copy()
    distances = {Strassen[source]: 0}
    for node in unvisited:
        distances[node] = float('inf') if not distances[node] else distances[node]

    while unvisited:
        current_node = min([edge[0] for edge in unvisited], key=distances.get)

        neighbors = [
            neighbor
            for neighbor in Strassen[current_node]
            if neighbor in unvisited
        ]

        for neighbor in neighbors:
            distance = distances[current_node] + Strassen[neighbor][2]  #save computing power to not calculate twice
            if distances[neighbor] and distances[neighbor] > distance:
                distances[neighbor] = distance

        unvisited.remove(current_node)

        if [Strassen[target][0] for target in targets] in distances.keys():
            continue

    return [dist[1] for dist in distances if dist[0] in targets]

def create_layered_graph(stages, strassen):

    adj_list = {} # {node: [[neighbor, distance]}

    stage = 1
    stage_to_node = {node[0]:node[1] for node in stages}

    current_stage = [
        node
        for node in stage_to_node
        if node[1] == stage]

    next_stage = [
        node
        for node in stage_to_node
        if node[1] == stage + 1]

    for cur in current_stage:
        next_ids = [node[0] for node in next_stage]
        distances = dijkstra(strassen, cur, next_ids)
        for nb in next_ids:
            adj_list[cur] = [nb, distances[nb]]



