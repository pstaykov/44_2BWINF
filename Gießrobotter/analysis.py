import sys
import os

# Ensure the parent directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import time
import matplotlib.pyplot as plt
from Aufgabe2 import evaluate_k

def analyze_roboter(file_path, output_json):
    """
    Analyze the computation time, longest path length, and average path length for different k values.

    Args:
        file_path (str): Path to the input file.
        output_json (str): Path to save the computed results as a JSON file.
    """
    # Load nodes from the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Skip the first two lines (max_len and num_nodes metadata)
        nodes = [tuple(map(float, line.strip().split())) for line in lines[2:]]

    num_nodes = len(nodes)
    max_len = float('inf')  # No constraint on max length for this analysis
    num_workers = 1  # Single-threaded for simplicity

    results = []

    for k in range(1, num_nodes // 2 + 1):  # Cap k at half the number of nodes
        start_time = time.time()
        max_distance, centroids, clusters, cluster_tours, improved_tours = evaluate_k(nodes, k, num_workers)
        end_time = time.time()

        computation_time = end_time - start_time
        average_path_length = sum(len(cluster) for cluster in clusters) / len(clusters)

        results.append({
            "k": k,
            "computation_time": computation_time,
            "longest_path_length": max_distance,
            "average_path_length": average_path_length
        })

    # Save results to a JSON file
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)

    return results

def plot_results(results):
    """
    Plot computation time and path lengths against k.

    Args:
        results (list): List of dictionaries containing k, computation time, longest path length, and average path length.
    """
    ks = [entry['k'] for entry in results]
    computation_times = [entry['computation_time'] for entry in results]
    longest_path_lengths = [entry['longest_path_length'] for entry in results]
    average_path_lengths = [entry['average_path_length'] for entry in results]

    # Plot computation time against k
    plt.figure(figsize=(10, 5))
    plt.plot(ks, computation_times, label='Computation Time', color='blue')
    plt.xlabel('k')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time vs k')
    plt.legend()
    plt.grid()
    plt.savefig('computation_time_vs_k.png')
    plt.show()

    # Plot longest and average path lengths against k
    plt.figure(figsize=(10, 5))
    plt.plot(ks, longest_path_lengths, label='Longest Path Length', color='red')
    plt.plot(ks, average_path_lengths, label='Average Path Length', color='green')
    plt.xlabel('k')
    plt.ylabel('Path Length')
    plt.title('Path Lengths vs k')
    plt.legend()
    plt.grid()
    plt.savefig('path_lengths_vs_k.png')
    plt.show()

if __name__ == "__main__":
    # File path for the input data
    input_file = os.path.join("A2_Gießroboter", "roboter04.txt")

    # Output JSON file to save results
    output_json = "analysis_results.json"

    # Perform analysis
    results = analyze_roboter(input_file, output_json)

    # Plot the results
    plot_results(results)
