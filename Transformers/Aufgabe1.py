import glob
import os
import itertools
from collections import defaultdict, deque

# Parsing / Rendering
def parse_dim(line: str):
    line = line.strip()
    parts = line.split()
    return int(parts[0]), int(parts[1])


def read_transform_file(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]

    # remove empty lines
    lines = [ln for ln in raw_lines if ln.strip() != ""]

    w1, h1 = parse_dim(lines[0])
    board1_lines = 2 * h1 - 1
    board1 = lines[1:1 + board1_lines]

    idx = 1 + board1_lines
    w2, h2 = parse_dim(lines[idx])
    board2_lines = 2 * h2 - 1
    board2 = lines[idx + 1: idx + 1 + board2_lines]

    return w1, h1, board1, w2, h2, board2


def board_to_parts(board_lines):
    nodes, verticals, horizontals = [], [], []
    id_counter = itertools.count(1)

    for y, line in enumerate(board_lines):
        for x, ch in enumerate(line.rstrip("\n")):
            if ch == "o":
                nodes.append({"id": next(id_counter), "type": "o", "x": x, "y": y})
            elif ch == "|":
                verticals.append({"id": next(id_counter), "type": "|", "x": x, "y": y})
            elif ch == "-":
                horizontals.append({"id": next(id_counter), "type": "-", "x": x, "y": y})

    return {"nodes": nodes, "verticals": verticals, "horizontals": horizontals}


def parts_to_board(parts, width, height):
    grid_w = 2 * width - 1
    grid_h = 2 * height - 1
    grid = [["." for _ in range(grid_w)] for _ in range(grid_h)]

    for node in parts["nodes"]:
        grid[node["y"]][node["x"]] = "o"
    for v in parts["verticals"]:
        grid[v["y"]][v["x"]] = "|"
    for h in parts["horizontals"]:
        grid[h["y"]][h["x"]] = "-"

    return ["".join(row) for row in grid]



# Normalization
def normalize_parts(parts):
    """Shift everything so min x,y becomes 0,0."""
    all_pieces = parts["nodes"] + parts["verticals"] + parts["horizontals"]
    if not all_pieces:
        return parts

    min_x = min(p["x"] for p in all_pieces)
    min_y = min(p["y"] for p in all_pieces)

    # Always shift by min
    for p in all_pieces:
        p["x"] -= min_x
        p["y"] -= min_y

    return parts


def shape_signature(parts):
    """
    Position-independent signature:
    compare exact coordinate sets per type.
    """
    tmp = {
        "nodes": [dict(p) for p in parts["nodes"]],
        "verticals": [dict(p) for p in parts["verticals"]],
        "horizontals": [dict(p) for p in parts["horizontals"]],
    }
    normalize_parts(tmp)

    nodes = frozenset((p["x"], p["y"]) for p in tmp["nodes"])
    verticals = frozenset((p["x"], p["y"]) for p in tmp["verticals"])
    horizontals = frozenset((p["x"], p["y"]) for p in tmp["horizontals"])
    return (nodes, verticals, horizontals)



# Connectivity Graph
def build_piece_index(parts):
    """id -> piece dict, and (x,y) -> piece ids by type"""
    by_id = {}
    at = defaultdict(list)
    for p in parts["nodes"] + parts["verticals"] + parts["horizontals"]:
        by_id[p["id"]] = p
        at[(p["x"], p["y"])].append(p["id"])
    return by_id, at


def build_adjacency(parts):
    """
    Build graph edges based on construction rules:
    - Node 'o' connects to adjacent sticks (up/down '|' and left/right '-')
    - Vertical '|' connects to nodes up/down
    - Horizontal '-' connects to nodes left/right
    returns:
    adjacency list: id -> set of connected ids
    """
    by_id, at = build_piece_index(parts)

    adj = defaultdict(set)

    def ids_at(x, y):
        return at.get((x, y), [])

    for pid, p in by_id.items():
        x, y, t = p["x"], p["y"], p["type"]

        if t == "o":
            # connect to vertical up/down
            for nx, ny in [(x, y - 1), (x, y + 1)]:
                for nid in ids_at(nx, ny):
                    if by_id[nid]["type"] == "|":
                        adj[pid].add(nid)
                        adj[nid].add(pid)
            # connect to horizontal left/right
            for nx, ny in [(x - 1, y), (x + 1, y)]:
                for nid in ids_at(nx, ny):
                    if by_id[nid]["type"] == "-":
                        adj[pid].add(nid)
                        adj[nid].add(pid)

        elif t == "|":
            # connect to nodes up/down
            for nx, ny in [(x, y - 1), (x, y + 1)]:
                for nid in ids_at(nx, ny):
                    if by_id[nid]["type"] == "o":
                        adj[pid].add(nid)
                        adj[nid].add(pid)

        elif t == "-":
            # connect to nodes left/right
            for nx, ny in [(x - 1, y), (x + 1, y)]:
                for nid in ids_at(nx, ny):
                    if by_id[nid]["type"] == "o":
                        adj[pid].add(nid)
                        adj[nid].add(pid)

    return adj

def BFS(adj, start):
    """
    Breadth-first search that works in stages (levels) and computes clusters.
    Returns the cluster of connected nodes starting from 'start' ending at every level of length 1.
    """

    distances = {start: 0}
    visited = {start}
    queue = deque([(start, 0)])

    for neighbor in adj[start]:
        distances[neighbor] = 1
        visited.add(neighbor)
        queue.append((neighbor, 1))

    while queue:
        node, level = queue.popleft()
        for neighbor in adj[node]:
            if neighbor not in visited:
                distances[neighbor] = level + 1
                visited.add(neighbor)
                queue.append((neighbor, level + 1))

    # check for stage count
    counts = {}

    for d in distances.values():
        counts[d] = counts.get(d, 0) + 1

    # filter counts to only include levels with count 1
    result = tuple(sorted(count for level, count in counts.items() if count == 1))
    return result


def find_permutations(parts):
    """
    Find all permutations of the adjacency matrix adj.
    Build every cluster for every node in the graph.
    combine every possible permutation of the adjacency matrix adj.
    """

    by_id, _ = build_piece_index(parts)
    adj = build_adjacency(parts)
    perms = set()

    for pid, piece in by_id.items():
        if piece['type'] == 'o':
            perms.add(BFS(adj, pid, by_id)[0])

    return perms
