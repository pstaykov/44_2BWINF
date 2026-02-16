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

def mirror_x(clust):
    '''
    Mirrors a cluster along the x-axis.
    '''
    maxx = max(clust, key=lambda x: x[0])[0]
    minx = min(clust, key=lambda x: x[0])[0]

    return [(maxx - x + minx, y) for x, y in clust]

def mirror_y(clust):
    '''
    Mirrors a cluster along the y-axis.
    '''
    maxy = max(clust, key=lambda x: x[1])[1]
    miny = min(clust, key=lambda x: x[1])[1]

    return [(x, maxy - y + miny) for x, y in clust]

def BFS(adj, start, end_node):
    """
    BFS that searches the subgraph "between" start and end_node.

    Behavior (matching the intended algorithm):
    - Work on a shallow copy of the adjacency mapping so original graph is untouched.
    - Remove the end_node from the graph (delete its edges).
    - Restrict start to only its first neighbor (keep that single edge); remove start from other neighbours.
    - Run BFS from the chosen neighbour so `start` is not included in the visited set.
    - If any traversal would discover `start` from another node (a back-edge to start), treat cluster as invalid -> return False.
    - Return set of visited node ids on success, or False if cluster invalid or not traversable.

    Notes:
    - `adj` is expected as a dict: node_id -> set(of neighbour ids).
    """
    # shallow copy of adjacency
    adj_copy = {nid: set(neis) for nid, neis in adj.items()}

    # If end_node exists, remove all edges to/from it
    if end_node in adj_copy:
        for neigh in list(adj_copy[end_node]):
            adj_copy[neigh].discard(end_node)
        adj_copy.pop(end_node, None)

    # start must exist and have at least one neighbour after removing end_node
    if start not in adj_copy or not adj_copy[start]:
        return False

    # choose deterministic first neighbour
    first_neigh = sorted(adj_copy[start])[0]

    # remove start from all other neighbours except first_neigh
    for neigh in list(adj_copy[start]):
        if neigh != first_neigh:
            adj_copy[neigh].discard(start)
    adj_copy[start] = {first_neigh}
    adj_copy.setdefault(first_neigh, set()).add(start)

    # BFS should start from the first neighbour so `start` is not included in the result
    visited = {first_neigh}
    queue = deque([first_neigh])

    while queue:
        node = queue.popleft()
        for neighbor in adj_copy.get(node, ()):
            # direct edge back to start -> invalid cluster
            if neighbor == start:
                return False
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited


def find_clusters(adj, parts, board_lines):
    """
    Find candidate clusters for mirroring.

    Returns a list of clusters, each cluster is a set of piece ids (integers).

    Strategy and assumptions:
    - Iterate over all node pieces (type 'o') as potential start nodes.
    - For each start node, consider other node pieces that lie on the same row (same y) or same column (same x)
      with strictly greater x / y (end candidates).
    - For each pair (start, end) call BFS(adj, start_id, end_id).
    - If BFS returns a visited set, and start_id is NOT inside the visited set (cluster is strictly between),
      we consider that visited set a candidate cluster.
    - The function uses the parts lists to map ids to coordinates.
    - If a detected cluster is a straight line, advance the start to the end node of that straight line
      (so the next start is positioned at the end of the straight segment).
    """
    # id to piece dict
    by_id = {}
    for p in parts["nodes"] + parts["verticals"] + parts["horizontals"]:
        by_id[p["id"]] = p

    # Collect node ids and their coords
    node_ids = [p["id"] for p in parts["nodes"]]
    coords = {nid: (by_id[nid]["x"], by_id[nid]["y"]) for nid in node_ids}

    clusters = []

    # For deterministic behavior sort node ids by coordinates
    sorted_nodes = sorted(node_ids, key=lambda nid: (coords[nid][1], coords[nid][0]))

    i = 0
    while i < len(sorted_nodes):
        start_id = sorted_nodes[i]
        sx, sy = coords[start_id]
        advanced = False

        # consider same-row and same-column end candidates with greater coordinate
        for j in range(i+1, len(sorted_nodes)):
            end_id = sorted_nodes[j]
            ex, ey = coords[end_id]
            # only same row (y) or same column (x) and strictly greater in the considered coordinate
            if not ((ex == sx and ey > sy) or (ey == sy and ex > sx)):
                continue

            visited = BFS(adj, start_id, end_id)
            if not visited or visited is False:
                continue

            # If start_id is visited, the cluster extends beyond the intended "between" area -> invalid
            if start_id in visited:
                continue

            # Heuristic check for trivial straight-line cluster:
            # count number of node pieces in the bounding interval along the axis
            if ex == sx:
                # vertical alignment: expected nodes count = (ey - sy) // 2 + 1 if grid spacing is 2
                expected_nodes = ((ey - sy) // 2) + 1
            else:
                expected_nodes = ((ex - sx) // 2) + 1

            cluster_node_count = sum(1 for nid in visited if by_id[nid]["type"] == "o")
            if cluster_node_count == expected_nodes:
                # Treat as a straight line; advance the start to the end of the straight line
                i = j
                advanced = True
                break

            clusters.append(visited)

        if not advanced:
            i += 1
        # if advanced is True, the loop continues with start at index i (set above)

    return clusters

def clone_parts(parts):
    """Return a deep copy of parts preserving ids."""
    return {"nodes": [dict(p) for p in parts["nodes"]],
            "verticals": [dict(p) for p in parts["verticals"]],
            "horizontals": [dict(p) for p in parts["horizontals"]]}


def apply_mirror_operation(parts, cluster_ids, axis_piece):
    """
    Return a new parts dict where every piece whose id is in cluster_ids
    is mirrored around the axis defined by axis_piece.
    axis_piece: piece dict of type '|' or '-'.
    For '|' (vertical stick) we mirror x: x' = 2*x_stab - x.
    For '-' (horizontal stick) we mirror y: y' = 2*y_stab - y.
    """
    newp = clone_parts(parts)
    # build id -> piece mapping for new copy
    byid = {p["id"]: p for p in (newp["nodes"] + newp["verticals"] + newp["horizontals"])}

    ax = axis_piece["x"]
    ay = axis_piece["y"]
    atype = axis_piece["type"]

    for cid in cluster_ids:
        if cid not in byid:
            continue
        piece = byid[cid]
        if atype == "|":
            piece["x"] = 2 * ax - piece["x"]
        elif atype == "-":
            piece["y"] = 2 * ay - piece["y"]
        else:
            # unsupported axis type -> no-op
            pass

    return newp


def mirror_whole(parts, axis):
    """Mirror the whole construction across its bounding box along axis 'x' or 'y'."""
    newp = clone_parts(parts)
    all_pieces = newp["nodes"] + newp["verticals"] + newp["horizontals"]
    if not all_pieces:
        return newp
    min_x = min(p["x"] for p in all_pieces)
    max_x = max(p["x"] for p in all_pieces)
    min_y = min(p["y"] for p in all_pieces)
    max_y = max(p["y"] for p in all_pieces)

    if axis == "x":
        for p in all_pieces:
            p["x"] = min_x + max_x - p["x"]
    elif axis == "y":
        for p in all_pieces:
            p["y"] = min_y + max_y - p["y"]
    return newp


def parts_signature(parts):
    """Helper: return normalized shape signature tuple for comparison."""
    tmp = clone_parts(parts)
    normalize_parts(tmp)
    return shape_signature(tmp)


def try_single_cluster_mirrors(src_parts, target_sig, adj):
    """Try mirroring each cluster once around every stick axis and compare to target_sig.
    Returns a list of successful operations (descriptions).
    """
    successes = []
    clusters = find_clusters(adj, src_parts, None)
    # build axis pieces list
    axes = src_parts["verticals"] + src_parts["horizontals"]
    # Build id->piece map for axis lookups
    all_by_id = {p["id"]: p for p in (src_parts["nodes"] + src_parts["verticals"] + src_parts["horizontals"])}

    seen_effects = set()

    for cl in clusters:
        for axis in axes:
            # only try axis types that make sense: '|' flips x, '-' flips y
            if axis["type"] == "|":
                axis_kind = "x"
            else:
                axis_kind = "y"

            newparts = apply_mirror_operation(src_parts, cl, axis)
            sig = parts_signature(newparts)
            if sig == target_sig:
                desc = f"mirror cluster(ids={sorted(cl)}) around {axis['type']} id={axis['id']} at ({axis['x']},{axis['y']})"
                if sig not in seen_effects:
                    successes.append(desc)
                    seen_effects.add(sig)
    return successes


def check_mirrors_and_report(src_parts, target_parts):
    """Check various mirror/symmetry options and return list of matching descriptions."""
    results = []
    target_sig = parts_signature(target_parts)
    src_sig = parts_signature(src_parts)

    if src_sig == target_sig:
        results.append("identical (no operation needed)")

    # full mirrors
    for ax in ("x", "y"):
        mp = mirror_whole(src_parts, ax)
        if parts_signature(mp) == target_sig:
            results.append(f"full mirror across {ax}")

    # adjacency required for cluster mirrors
    adj = build_adjacency(src_parts)
    single = try_single_cluster_mirrors(src_parts, target_sig, adj)
    results.extend(single)

    return results


def process_file(path):
    w1, h1, board1, w2, h2, board2 = read_transform_file(path)
    src_parts = board_to_parts(board1)
    tgt_parts = board_to_parts(board2)

    res = check_mirrors_and_report(src_parts, tgt_parts)

    print(f"File: {os.path.basename(path)} -> ", end="")
    if res:
        print("MATCH: ")
        for r in res:
            print(f"  - {r}")
    else:
        print("NO MATCH FOUND")


def main():
    """Main loop: iterate over all .txt in Transformers/eingaben and process them."""
    base = os.path.join(os.path.dirname(__file__), "eingaben")
    pattern = os.path.join(base, "*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No input files found in Transformers/eingaben")
        return
    for f in files:
        process_file(f)


if __name__ == '__main__':
    main()
