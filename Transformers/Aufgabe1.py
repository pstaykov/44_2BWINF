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

    # Always shift by min (even if already 0)
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



# Connectivity Graph (correct!)
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
    Build graph edges based on actual construction rules:
    - Node 'o' connects to adjacent sticks (up/down '|' and left/right '-')
    - Vertical '|' connects to nodes up/down
    - Horizontal '-' connects to nodes left/right
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


def endpoints_of_stick(stick_piece, parts):
    """Return the two endpoint node IDs of a stick (if present)."""
    by_id, at = build_piece_index(parts)
    x, y, t = stick_piece["x"], stick_piece["y"], stick_piece["type"]

    def node_at(nx, ny):
        for pid in at.get((nx, ny), []):
            if by_id[pid]["type"] == "o":
                return pid
        return None

    if t == "|":
        return node_at(x, y - 1), node_at(x, y + 1)
    if t == "-":
        return node_at(x - 1, y), node_at(x + 1, y)
    return (None, None)


def component_from(start_id, adj, blocked_edge=None):
    """
    BFS component starting from start_id.
    blocked_edge: tuple(u,v) meaning that single edge is removed.
    """
    if start_id is None:
        return set()

    u_block, v_block = blocked_edge if blocked_edge else (None, None)

    seen = set([start_id])
    q = deque([start_id])
    while q:
        u = q.popleft()
        for v in adj[u]:
            # skip the blocked edge in both directions
            if blocked_edge and ((u == u_block and v == v_block) or (u == v_block and v == u_block)):
                continue
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


# Mirroring Operations
def mirror_cluster(parts, stick_id, which_side):
    """
    Apply mirroring around the axis defined by stick_id.
    which_side: 0 or 1 (mirror the component of endpoint0 or endpoint1)
    """
    # Build fast access
    by_id, _ = build_piece_index(parts)
    adj = build_adjacency(parts)

    stick = by_id[stick_id]
    if stick["type"] not in ("|", "-"):
        return None  # invalid

    e0, e1 = endpoints_of_stick(stick, parts)
    if e0 is None or e1 is None:
        return None  # malformed input


    # To "split" clusters at the hinge, we remove the stick node from traversal.

    adj2 = defaultdict(set)
    for u, nbrs in adj.items():
        if u == stick_id:
            continue
        for v in nbrs:
            if v == stick_id:
                continue
            adj2[u].add(v)

    comp0 = component_from(e0, adj2)
    comp1 = component_from(e1, adj2)

    # If hinge doesn't actually separate into two different comps, skip (no meaningful fold)
    if comp0 == comp1:
        return None

    moved = comp0 if which_side == 0 else comp1

    # Mirror coordinates of all moved pieces (including nodes + sticks) around axis through the stick
    axis_x = stick["x"]
    axis_y = stick["y"]

    # Deep copy pieces list
    new_parts = {
        "nodes": [dict(p) for p in parts["nodes"]],
        "verticals": [dict(p) for p in parts["verticals"]],
        "horizontals": [dict(p) for p in parts["horizontals"]],
    }

    # Build id->piece reference in new_parts for mutation
    new_by_id, _ = build_piece_index(new_parts)

    if stick["type"] == "|":
        # vertical axis: reflect x
        for pid in moved:
            p = new_by_id.get(pid)
            if p is None:
                continue
            p["x"] = 2 * axis_x - p["x"]
    else:
        # horizontal axis: reflect y
        for pid in moved:
            p = new_by_id.get(pid)
            if p is None:
                continue
            p["y"] = 2 * axis_y - p["y"]

    normalize_parts(new_parts)
    return new_parts


def is_symmetric_cluster(parts, stick_id, which_side):
    """
    If mirroring that side yields no change in normalized shape, it's symmetric.
    """
    before = shape_signature(parts)
    after_parts = mirror_cluster(parts, stick_id, which_side)
    if after_parts is None:
        return True  # treat as not a valid permutation
    after = shape_signature(after_parts)
    return before == after


# Generate possible permutations
def generate_permutations(parts):
    """
    Returns a list of operations:
    op = (stick_id, which_side)
    Only include non-symmetric operations; deduplicate identical ones.
    """
    all_sticks = parts["verticals"] + parts["horizontals"]
    perms = []
    seen = set()

    for stick in all_sticks:
        sid = stick["id"]
        for which_side in (0, 1):
            if is_symmetric_cluster(parts, sid, which_side):
                continue

            # Deduplicate by the effect on normalized shape from the ORIGINAL parts
            result = mirror_cluster(parts, sid, which_side)
            if result is None:
                continue

            key = (shape_signature(result))
            if key in seen:
                continue
            seen.add(key)
            perms.append((sid, which_side))

    return perms



# Backtracking search (each perm at most once)
def find_chain_to_target(given_parts, target_sig, perms, max_depth=None):
    """
    Try sequences of permutations (order matters, each used once) via backtracking.
    Prints and returns the found chain, or None.
    """
    if max_depth is None:
        max_depth = len(perms)

    # pruning: remember visited (signature, used_mask) or just signature at depth
    visited = set()

    def dfs(current_parts, start_idx, chain, used):
        sig = shape_signature(current_parts)
        if sig == target_sig:
            return chain

        if len(chain) >= max_depth:
            return None

        # strong prune: don't revisit same signature with same used set size
        state_key = (sig, tuple(sorted(used)))
        if state_key in visited:
            return None
        visited.add(state_key)

        for i, (sid, which_side) in enumerate(perms):
            if i in used:
                continue
            nxt = mirror_cluster(current_parts, sid, which_side)
            if nxt is None:
                continue
            used.add(i)
            chain.append((sid, which_side))
            res = dfs(nxt, i + 1, chain, used)
            if res is not None:
                return res
            chain.pop()
            used.remove(i)

        return None

    return dfs(given_parts, 0, [], set())


# Main
def main():
    folder = "eingaben"
    pattern = os.path.join(folder, "transform*.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found: {pattern}")
        return

    for path in files:
        w1, h1, given_board, w2, h2, target_board = read_transform_file(path)

        given_parts = board_to_parts(given_board)
        target_parts = board_to_parts(target_board)

        # Normalize both for position independence
        normalize_parts(given_parts)
        normalize_parts(target_parts)

        target_sig = shape_signature(target_parts)

        perms = generate_permutations(given_parts)

        print(f"\n{os.path.basename(path)}")
        print(f"Possible permutations (deduped): {len(perms)}")

        chain = find_chain_to_target(given_parts, target_sig, perms, max_depth=len(perms))

        if chain is None:
            print("NO")
        else:
            print("YES")
            print("Chain (stick_id, side):")
            # side 0 mirrors endpoint0-component, side 1 mirrors endpoint1-component
            for sid, side in chain:
                print(f"  ({sid}, {side})")

            # Optionally show final board
            cur = given_parts
            for sid, side in chain:
                cur = mirror_cluster(cur, sid, side)
            # Render with some width/height guess:
            # (safe render size based on max coords)
            all_pieces = cur["nodes"] + cur["verticals"] + cur["horizontals"]
            max_x = max(p["x"] for p in all_pieces) if all_pieces else 0
            max_y = max(p["y"] for p in all_pieces) if all_pieces else 0
            # Convert grid coords back to "board dims"
            out_w = (max_x // 2) + 1
            out_h = (max_y // 2) + 1
            print("Final (normalized) result:")
            for ln in parts_to_board(cur, out_w, out_h):
                print(ln)


if __name__ == "__main__":
    main()
