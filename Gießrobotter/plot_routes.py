"""Plot routes from existing input and output files.

This script does NOT compute routes. It reads inputs from A2_Gießroboter/ and
matching outputs from outputs/ (files named <inputname>_out.txt) and plots the
stations and visited tree sequences for each robot.

Usage:
    python plot_routes.py            # loop over all input files that have outputs
    python plot_routes.py <inputfile> # plot only the specified input file (filename or path)
"""
import os
import sys

# Prefer seaborn for nicer palettes; fall back to matplotlib
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    _HAS_SEABORN = True
except Exception:
    try:
        import matplotlib.pyplot as plt
        _HAS_SEABORN = False
    except Exception:
        plt = None
        _HAS_SEABORN = False

# reuse the input parsing from Aufgabe2 if available for consistency
try:
    from Aufgabe2 import read_file
except Exception:
    # provide a local read_file fallback (same logic)
    def read_file(path: str):
        with open(path, "r", encoding="utf-8") as f:
            raw_lines = [ln.strip() for ln in f.readlines() if ln.strip() != ""]

        coord_start = None
        for i, ln in enumerate(raw_lines):
            parts = ln.split()
            if len(parts) >= 3:
                coord_start = i
                break
        if coord_start is None:
            raise ValueError("No coordinate lines found in input file")
        if coord_start == 0:
            max_len = float(raw_lines[0])
            start_idx = 1
        else:
            max_len = float(raw_lines[coord_start - 1])
            start_idx = coord_start
        nodes = []
        for ln in raw_lines[start_idx:]:
            parts = ln.split()
            if len(parts) < 3:
                continue
            try:
                id_ = int(parts[0])
                x = int(parts[1])
                y = int(parts[2])
            except ValueError:
                continue
            nodes.append((id_, x, y))
        return nodes, max_len


def read_output(path: str):
    """Parse an output file produced by Aufgabe2.

    Returns:
      k: int
      max_len: float (as recorded)
      routes: list of dicts { 'length': float, 'station': (x,y), 'ids': [int,...] }
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.readlines() if ln.strip() != ""]

    if not lines:
        raise ValueError("Empty output file")

    # first line: k
    k = int(lines[0].split()[0])
    # second line: max_len
    max_len = float(lines[1].split()[0]) if len(lines) >= 2 else 0.0

    routes = []
    idx = 2
    while idx < len(lines):
        # three lines per route: length, station, ids
        length_line = lines[idx] if idx < len(lines) else "0"
        try:
            length = float(length_line.split()[0])
        except Exception:
            length = 0.0
        idx += 1

        station = (0, 0)
        if idx < len(lines):
            st_parts = lines[idx].split()
            if len(st_parts) >= 2:
                try:
                    station = (int(st_parts[0]), int(st_parts[1]))
                except Exception:
                    station = (0, 0)
        idx += 1

        ids = []
        if idx < len(lines):
            id_parts = lines[idx].split()
            try:
                ids = [int(x) for x in id_parts]
            except Exception:
                ids = []
        idx += 1

        routes.append({'length': length, 'station': station, 'ids': ids})

    return k, max_len, routes


def build_palette(n):
    n = max(1, n)
    try:
        if _HAS_SEABORN:
            sns.set(style="whitegrid")
            palette = list(sns.color_palette(n_colors=n))
        else:
            cmap_name = 'tab20' if n <= 20 else 'hsv'
            cmap = plt.cm.get_cmap(cmap_name)
            palette = [cmap(i / n) for i in range(n)]
    except Exception:
        cmap = plt.cm.get_cmap('hsv')
        palette = [cmap(i / n) for i in range(n)]
    return palette


def plot_pair(input_path, output_path, show=True):
    if plt is None:
        print("matplotlib not available; cannot plot")
        return

    nodes, _ = read_file(input_path)
    id_to_coord = {nid: (x, y) for (nid, x, y) in nodes}

    k, max_len, routes = read_output(output_path)

    # prepare palette
    palette = build_palette(len(routes))

    fig, ax = plt.subplots(figsize=(8, 8))

    # plot all trees
    xs = [x for (_id, x, y) in nodes]
    ys = [y for (_id, x, y) in nodes]
    ax.scatter(xs, ys, c='k', s=20)

    # annotate ids
    for nid, (x, y) in id_to_coord.items():
        ax.text(x + 0.2, y + 0.2, str(nid), fontsize=8, color='gray')

    # plot each route in its own unique color
    for i, r in enumerate(routes):
        color = palette[i]
        station = r.get('station')
        ids = r.get('ids', [])
        route_coords = []
        for rid in ids:
            if rid in id_to_coord:
                route_coords.append(id_to_coord[rid])
            else:
                # id not found in input; skip
                pass

        # plot station
        if station is not None:
            ax.scatter([station[0]], [station[1]], c=[color], marker='s', s=80, edgecolors='black', label=f"Station {i+1}")

        if route_coords:
            xvals = [p[0] for p in route_coords]
            yvals = [p[1] for p in route_coords]
            # connect station to route start
            if station is not None:
                xvals = [station[0]] + xvals
                yvals = [station[1]] + yvals
            ax.plot(xvals, yvals, color=color, linewidth=2, alpha=0.85)
            ax.scatter([p[0] for p in route_coords], [p[1] for p in route_coords], c=[color], s=40)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Routes for {os.path.basename(input_path)} (k={k})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()

    if show:
        print(f"Displaying plot for {os.path.basename(input_path)}. Close window to continue.")
        plt.show()
    else:
        # return figure for further programmatic handling
        return fig, ax


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "A2_Gießroboter")
    outputs_dir = os.path.join(script_dir, "outputs")

    args = sys.argv[1:]
    single = None
    if args:
        single = args[0]

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.txt')])

    for fname in files:
        if fname.endswith('_out.txt'):
            continue
        if single:
            # allow either basename or path
            if os.path.abspath(single) != os.path.abspath(os.path.join(input_dir, fname)) and os.path.basename(single) != fname:
                continue

        input_path = os.path.join(input_dir, fname)
        out_name = fname + '_out.txt'
        output_path = os.path.join(outputs_dir, out_name)
        if not os.path.exists(output_path):
            print(f"Skipping {fname}: output not found at {output_path}")
            continue

        try:
            plot_pair(input_path, output_path, show=True)
        except Exception as e:
            print(f"Failed plotting {fname}: {e}")


if __name__ == '__main__':
    main()
