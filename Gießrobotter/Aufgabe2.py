import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]
    
    max_len = raw_lines[0]
    nodes = [ln.split() for ln in raw_lines[2:]] # id x, y, centroid

    return nodes, max_len

def plot_graph(nodes):
    x = [float(node[1]) for node in nodes]
    y = [float(node[2]) for node in nodes]
    
    plt.scatter(x, y)
    
    for node in nodes:
        plt.text(float(node[1]), float(node[2]), node[0])
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Graph Nodes')
    plt.grid(True)
    plt.show()

def k_means(nodes, k, iter=100):
    # Initialize centroids randomly from existing nodes
    centroids = random.sample([(float(node[1]), float(node[2])) for node in nodes], k)

    for _ in range(iter):
        clusters = [[] for _ in range(k)]
        
        # Assign nodes to the nearest centroid
        for node in nodes:
            point = (float(node[1]), float(node[2]))
            distances = [np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(point)
        
        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:  # Avoid division by zero
                new_centroid = (sum(p[0] for p in cluster) / len(cluster),
                                sum(p[1] for p in cluster) / len(cluster))
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.choice([(float(node[1]), float(node[2])) for node in nodes]))
        
        if new_centroids == centroids:
            break  # Convergence
        centroids = new_centroids

    return centroids, clusters

def plot_graph_with_clusters(nodes, centroids, clusters):
    colors = sns.color_palette("hsv", len(clusters))
    
    for idx, cluster in enumerate(clusters):
        x = [point[0] for point in cluster]
        y = [point[1] for point in cluster]
        plt.scatter(x, y, color=colors[idx], label=f'Cluster {idx+1}')
    
    # Plot centroids
    centroid_x = [centroid[0] for centroid in centroids]
    centroid_y = [centroid[1] for centroid in centroids]
    plt.scatter(centroid_x, centroid_y, color='black', marker='X', s=100, label='Centroids')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Graph Nodes with Clusters')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    path = "A2_Gießroboter/roboter01.txt"
    nodes, max_len = read_file(path)
    print(nodes)
    plot_graph(nodes)
    k = 3  # Number of clusters
    centroids, clusters = k_means(nodes, k)
    plot_graph_with_clusters(nodes, centroids, clusters)
    
