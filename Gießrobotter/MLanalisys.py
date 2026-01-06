import numpy as np
import matplotlib.pyplot as plt

# Use our simple standard-library LinearRegression when available
try:
    from Gießrobotter.MLalgo import LinearRegression
except Exception:
    # fallback: relative import for direct script execution
    from .MLalgo import LinearRegression


def scatter_and_fit(results, save_path=None):
    """Scatterplot longest_path_length vs average_cluster_size and fit a simple linear regression.

    Before fitting/plotting, this function permanently removes the two datapoints
    with the highest longest_path_length values (outliers) from the data used
    for both plotting and fitting.

    Args:
        results (list of dict): each dict should contain 'longest_path_length' and 'average_cluster_size'
        save_path (str): optional path to save the figure

    Returns:
        slope (float), intercept (float), mse (float)
    """
    if not results:
        raise ValueError("No results provided to scatter_and_fit")

    # Extract arrays
    x = np.array([r.get('average_cluster_size', 0) for r in results], dtype=float)
    y = np.array([r.get('longest_path_length', 0) for r in results], dtype=float)

    # Filter out any NaNs or infinite values
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if x.size == 0:
        raise ValueError("No finite data available for fitting")

    # Remove the two datapoints with the highest longest_path_length (y) values, if possible
    if x.size > 2:
        # argsort descending on y
        desc_idx = np.argsort(y)[::-1]
        # indices to remove (top two largest y)
        remove_idx = desc_idx[:2]
        keep_mask = np.ones_like(y, dtype=bool)
        keep_mask[remove_idx] = False
        x = x[keep_mask]
        y = y[keep_mask]
    else:
        # Not enough points to remove; keep all
        pass

    if x.size == 0:
        raise ValueError("No data left after removing top outliers")

    # Fit simple linear regression y = m*x + b using our stdlib LinearRegression
    X = [[float(v)] for v in x.tolist()]
    model = LinearRegression()
    model.fit(X, y.tolist())
    m = model.coef_[0]
    b = model.intercept_

    # Predictions and MSE
    # compute predictions directly from coef_ and intercept_ (no predict method)
    y_pred = np.array([model.intercept_ + sum(c * xi for c, xi in zip(model.coef_, row)) for row in X], dtype=float)
    mse = float(np.mean((y - y_pred) ** 2))

    # Plot scatter and regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, label='Data points (outliers removed)')
    # Plot regression line over the span of x
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Fit: y={m:.3f}x+{b:.3f}')

    plt.xlabel('Average cluster size (number of datapoints)')
    plt.ylabel('Longest path length')
    plt.title('Longest path length vs Average cluster size (top-2 y removed)')
    plt.legend()
    plt.grid(True)

    # Annotate with MSE and model params
    plt.annotate(f'MSE={mse:.4f}\n slope={m:.4f}\n intercept={b:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved scatter+fit plot to: {save_path}")
        except Exception as e:
            print(f"Failed to save scatter+fit plot to {save_path}: {e}")

    plt.show()

    return float(m), float(b), mse


def plot_avg_cluster_size_vs_k(results, save_path='avgcluster_vs_k.png'):
    """Scatter average cluster size against number of clusters (k).

    Expects each entry in results to contain 'k' and 'average_cluster_size'.
    Returns (slope, intercept, mse) for the fitted line or (None, None, None) if fitting wasn't possible.
    """
    if not results:
        print("No results provided for plot_avg_cluster_size_vs_k")
        return None, None, None

    ks = np.array([r.get('k', np.nan) for r in results], dtype=float)
    avg_sizes = np.array([r.get('average_cluster_size', np.nan) for r in results], dtype=float)

    mask = np.isfinite(ks) & np.isfinite(avg_sizes)
    ks = ks[mask]
    avg_sizes = avg_sizes[mask]

    if ks.size == 0:
        print("No finite data for avg cluster size vs k")
        return None, None, None

    # Fit linear model avg_sizes = m * ks + b if enough points
    if ks.size >= 2:
        X = [[float(v)] for v in ks.tolist()]
        model = LinearRegression()
        model.fit(X, avg_sizes.tolist())
        m = model.coef_[0]
        b = model.intercept_
        y_pred = np.array([model.intercept_ + sum(c * xi for c, xi in zip(model.coef_, row)) for row in X], dtype=float)
        mse = float(np.mean((avg_sizes - y_pred) ** 2))
    else:
        m = None
        b = None
        mse = None

    plt.figure(figsize=(8, 6))
    plt.scatter(ks, avg_sizes, alpha=0.7)

    if m is not None:
        x_line = np.linspace(ks.min(), ks.max(), 100)
        plt.plot(x_line, m * x_line + b, color='red', label=f'Fit: y={m:.3f}x+{b:.3f}')
        plt.legend()

    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Average cluster size')
    plt.title('Average cluster size vs Number of clusters (k)')
    plt.grid(True)

    if m is not None:
        plt.annotate(f'MSE={mse:.4f}\n slope={m:.4f}\n intercept={b:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {save_path}")
        except Exception as e:
            print(f"Failed to save {save_path}: {e}")

    plt.show()

    return (float(m) if m is not None else None,
            float(b) if b is not None else None,
            float(mse) if mse is not None else None)


def plot_avg_vs_max_cluster_size(results, save_path='avg_vs_max_cluster_size.png'):
    """Scatter average cluster size against max cluster size.

    Expects 'average_cluster_size' and 'max_cluster_size' in each results entry.
    Returns (slope, intercept, mse) for the fitted line or (None, None, None).
    """
    if not results:
        print("No results provided for plot_avg_vs_max_cluster_size")
        return None, None, None

    avg_sizes = np.array([r.get('average_cluster_size', np.nan) for r in results], dtype=float)
    max_sizes = np.array([r.get('max_cluster_size', np.nan) for r in results], dtype=float)

    mask = np.isfinite(avg_sizes) & np.isfinite(max_sizes)
    avg_sizes = avg_sizes[mask]
    max_sizes = max_sizes[mask]

    if avg_sizes.size == 0:
        print("No finite data for average vs max cluster size")
        return None, None, None

    # Fit linear model max_sizes = m * avg_sizes + b if enough points
    if avg_sizes.size >= 2:
        X = [[float(v)] for v in avg_sizes.tolist()]
        model = LinearRegression()
        model.fit(X, max_sizes.tolist())
        m = model.coef_[0]
        b = model.intercept_
        y_pred = np.array([model.intercept_ + sum(c * xi for c, xi in zip(model.coef_, row)) for row in X], dtype=float)
        mse = float(np.mean((max_sizes - y_pred) ** 2))
    else:
        m = None
        b = None
        mse = None

    plt.figure(figsize=(8, 6))
    plt.scatter(avg_sizes, max_sizes, alpha=0.7)

    if m is not None:
        x_line = np.linspace(avg_sizes.min(), avg_sizes.max(), 100)
        plt.plot(x_line, m * x_line + b, color='red', label=f'Fit: y={m:.3f}x+{b:.3f}')
        plt.legend()

    plt.xlabel('Average cluster size')
    plt.ylabel('Max cluster size')
    plt.title('Average cluster size vs Max cluster size')
    plt.grid(True)

    if m is not None:
        plt.annotate(f'MSE={mse:.4f}\n slope={m:.4f}\n intercept={b:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {save_path}")
        except Exception as e:
            print(f"Failed to save {save_path}: {e}")

    plt.show()

    return (float(m) if m is not None else None,
            float(b) if b is not None else None,
            float(mse) if mse is not None else None)


def plot_max_cluster_size_vs_max_path(results, save_path='maxcluster_vs_maxpath.png'):
    """Scatter max cluster size against longest path length.

    Expects 'max_cluster_size' and 'longest_path_length' in each results entry.
    Returns (slope, intercept, mse) for the fitted line or (None, None, None).
    """
    if not results:
        print("No results provided for plot_max_cluster_size_vs_max_path")
        return None, None, None

    max_sizes = np.array([r.get('max_cluster_size', np.nan) for r in results], dtype=float)
    longest_paths = np.array([r.get('longest_path_length', np.nan) for r in results], dtype=float)

    mask = np.isfinite(max_sizes) & np.isfinite(longest_paths)
    max_sizes = max_sizes[mask]
    longest_paths = longest_paths[mask]

    if max_sizes.size == 0:
        print("No finite data for max cluster size vs longest path length")
        return None, None, None

    # Fit linear model longest_paths = m * max_sizes + b if enough points
    if max_sizes.size >= 2:
        X = [[float(v)] for v in max_sizes.tolist()]
        model = LinearRegression()
        model.fit(X, longest_paths.tolist())
        m = model.coef_[0]
        b = model.intercept_
        y_pred = np.array([model.intercept_ + sum(c * xi for c, xi in zip(model.coef_, row)) for row in X], dtype=float)
        mse = float(np.mean((longest_paths - y_pred) ** 2))
    else:
        m = None
        b = None
        mse = None

    plt.figure(figsize=(8, 6))
    plt.scatter(max_sizes, longest_paths, alpha=0.7)

    if m is not None:
        x_line = np.linspace(max_sizes.min(), max_sizes.max(), 100)
        plt.plot(x_line, m * x_line + b, color='red', label=f'Fit: y={m:.3f}x+{b:.3f}')
        plt.legend()

    plt.xlabel('Max cluster size')
    plt.ylabel('Longest path length')
    plt.title('Max cluster size vs Longest path length')
    plt.grid(True)

    if m is not None:
        plt.annotate(f'MSE={mse:.4f}\n slope={m:.4f}\n intercept={b:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {save_path}")
        except Exception as e:
            print(f"Failed to save {save_path}: {e}")

    plt.show()

    return (float(m) if m is not None else None,
            float(b) if b is not None else None,
            float(mse) if mse is not None else None)


def create_three_scatter_plots(results, save_prefix='scatter'):
    """Convenience function to create the three requested scatter plots and return their file paths."""
    p1 = f"{save_prefix}_avgcluster_vs_k.png"
    p2 = f"{save_prefix}_avg_vs_maxcluster.png"
    p3 = f"{save_prefix}_maxcluster_vs_maxpath.png"

    m1 = plot_avg_cluster_size_vs_k(results, save_path=p1)
    m2 = plot_avg_vs_max_cluster_size(results, save_path=p2)
    m3 = plot_max_cluster_size_vs_max_path(results, save_path=p3)

    params = {
        p1: m1,
        p2: m2,
        p3: m3,
    }

    return p1, p2, p3, params


def plot_learned_parameters(params, save_path='learned_parameters.png'):
    """Plot the learned parameters (slope, intercept, MSE) for each generated scatter plot.

    params: dict mapping plot_filename -> (slope, intercept, mse)
    Returns a dict mapping plot_filename -> (slope, intercept, mse) with NaN for missing values.
    """
    if not params:
        print("No parameters provided to plot_learned_parameters")
        return {}

    names = list(params.keys())
    slopes = [params[n][0] if params[n] and params[n][0] is not None else np.nan for n in names]
    intercepts = [params[n][1] if params[n] and params[n][1] is not None else np.nan for n in names]
    mses = [params[n][2] if params[n] and params[n][2] is not None else np.nan for n in names]

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].bar(names, slopes, color='tab:blue')
    axs[0].set_title('Learned slopes')
    axs[0].set_ylabel('slope')

    axs[1].bar(names, intercepts, color='tab:orange')
    axs[1].set_title('Learned intercepts')
    axs[1].set_ylabel('intercept')

    axs[2].bar(names, mses, color='tab:green')
    axs[2].set_title('MSE')
    axs[2].set_ylabel('MSE')

    for ax in axs:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(True)

    fig.tight_layout()

    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved learned parameters plot to: {save_path}")
        except Exception as e:
            print(f"Failed to save learned parameters plot to {save_path}: {e}")

    plt.show()

    # Return a cleaned dict
    cleaned = {n: (float(slopes[i]) if not np.isnan(slopes[i]) else None,
                   float(intercepts[i]) if not np.isnan(intercepts[i]) else None,
                   float(mses[i]) if not np.isnan(mses[i]) else None)
               for i, n in enumerate(names)}
    return cleaned
