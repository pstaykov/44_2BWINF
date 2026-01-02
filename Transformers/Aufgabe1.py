import glob
import os


def parse_dim(line):
    """Parses 'W H' (ignores optional trailing comments like '# ...')."""
    line = line.strip()
    parts = line.split()
    return int(parts[0]), int(parts[1])


def board_to_parts(board_lines):
    """Convert ASCII board lines into 3 sets of coordinates."""
    nodes = set()
    verticals = set()
    horizontals = set()

    for y, line in enumerate(board_lines):
        for x, ch in enumerate(line.rstrip("\n")):
            if ch == "o":
                nodes.add((x, y))
            elif ch == "|":
                verticals.add((x, y))
            elif ch == "-":
                horizontals.add((x, y))

    return {"nodes": nodes, "verticals": verticals, "horizontals": horizontals}


def parts_to_board(parts, width, height):
    """Render parts into ASCII board lines."""
    grid_w = 2 * width - 1
    grid_h = 2 * height - 1
    grid = [["." for _ in range(grid_w)] for _ in range(grid_h)]

    for x, y in parts["nodes"]:
        grid[y][x] = "o"
    for x, y in parts["verticals"]:
        grid[y][x] = "|"
    for x, y in parts["horizontals"]:
        grid[y][x] = "-"

    return ["".join(row) for row in grid]


def mirror_horizontal(parts, width, height):
    """
    Mirror along vertical axis in ASCII grid.
    (x,y) = (ascii_w-1-x, y)
    """
    ascii_w = 2 * width - 1

    def mx(c):
        x, y = c
        return (ascii_w - 1 - x, y)

    return {k: {mx(c) for c in v} for k, v in parts.items()}


def mirror_vertical(parts, width, height):
    """
    Mirror along horizontal axis in ASCII grid.
    (x,y) = (x, ascii_h-1-y)
    """
    ascii_h = 2 * height - 1

    def my(c):
        x, y = c
        return (x, ascii_h - 1 - y)

    return {k: {my(c) for c in v} for k, v in parts.items()}


def all_permutations(parts, width, height):
    """Original, horizontal mirror, vertical mirror, both."""
    h = mirror_horizontal(parts, width, height)
    v = mirror_vertical(parts, width, height)
    hv = mirror_horizontal(v, width, height)  # == mirror_vertical(h, width, height)
    return [parts, h, v, hv]


def same_construction(a, b):
    return (
        a["nodes"] == b["nodes"]
        and a["verticals"] == b["verticals"]
        and a["horizontals"] == b["horizontals"]
    )


def read_transform_file(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]

    # remove purely empty lines between blocks
    lines = [ln for ln in raw_lines if ln.strip() != ""]

    # First board (ASCII grid has 2*h-1 rows)
    w1, h1 = parse_dim(lines[0])
    board1_lines = 2 * h1 - 1
    board1 = lines[1 : 1 + board1_lines]
    # Second board
    idx = 1 + board1_lines
    w2, h2 = parse_dim(lines[idx])
    board2_lines = 2 * h2 - 1
    board2 = lines[idx + 1 : idx + 1 + board2_lines]

    return w1, h1, board1, w2, h2, board2


def can_obtain_by_mirroring(given_lines, target_lines, width, height):
    given_parts = board_to_parts(given_lines)
    target_parts = board_to_parts(target_lines)

    for p in all_permutations(given_parts, width, height):
        if same_construction(p, target_parts):
            return True
    return False


def main():
    folder = "eingaben"
    pattern = os.path.join(folder, "transform*.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found: {pattern}")
        return

    for path in files:
        w1, h1, given_board, w2, h2, target_board = read_transform_file(path)

        print(f"{os.path.basename(path)}")
        print("given:")
        for line in given_board:
            print(line)
        print("target:")
        for line in target_board:
            print(line)

        print("permutations:")
        given_parts = board_to_parts(given_board)
        for idx, p in enumerate(all_permutations(given_parts, width=w1, height=h1), start=1):
            print(f"perm {idx}:")
            for line in parts_to_board(p, width=w1, height=h1):
                print(line)

        ok = can_obtain_by_mirroring(given_board, target_board, width=w1, height=h1)
        print(f"{os.path.basename(path)}: {'YES' if ok else 'NO'}")


if __name__ == "__main__":
    main()
