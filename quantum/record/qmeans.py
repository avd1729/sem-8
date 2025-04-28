import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
import pandas as pd
from scipy.spatial import distance

np.random.seed(42)

dataset = load_wine()
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names
class_names = dataset.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

min_max_scaler = MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(X_pca)

def encode_data_point(data_point, qc, qubits):
    """Encode a data point into a quantum state using amplitude encoding."""
    theta_x = data_point[0] * np.pi
    theta_y = data_point[1] * np.pi
    
    qc.ry(theta_x, qubits[0])
    qc.ry(theta_y, qubits[1])

def quantum_distance(point1, point2, num_shots=1024):
    """Calculate distance between two points using quantum circuit."""
    qc = QuantumCircuit(3, 1)
    
    encode_data_point(point1, qc, [0, 1])
    
    qc.x(2)
    qc.h(2)
    
    qc.cx(0, 2)
    qc.cx(1, 2)
    
    qc.ry(-point2[0] * np.pi, 0)
    qc.ry(-point2[1] * np.pi, 1)
    
    qc.h(2)
    
    qc.measure(2, 0)
    
    backend = FakeManilaV2()
    
    transpiled_circuit = transpile(qc, backend)
    
    try:
        job = backend.run(transpiled_circuit, shots=num_shots)
        result = job.result()
        counts = result.get_counts()
        
        prob_1 = counts.get('1', 0) / num_shots
    except:
        print("Warning: Quantum simulation failed, using classical distance as fallback")
        prob_1 = np.sum((point1 - point2) ** 2) / 4  
    
    distance_value = np.sqrt(prob_1)
    
    return distance_value

def classical_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

class QKMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, use_quantum=True):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.use_quantum = use_quantum
        
    def initialize_centroids(self, X):
        """Initialize centroids using the k-means++ method."""
        centroids = [X[np.random.choice(X.shape[0])]]
        
        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min([np.sum((x-c)**2) for c in centroids]) for x in X])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids.append(X[j])
                    break
                    
        return np.array(centroids)
    
    def assign_clusters(self, X):
        """Assign each point to the nearest centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i in range(X.shape[0]):
            for j in range(self.n_clusters):
                if self.use_quantum:
                    try:
                        distances[i, j] = quantum_distance(X[i], self.centroids[j])
                    except Exception as e:
                        print(f"Quantum calculation failed with error: {e}")
                        print("Falling back to classical distance")
                        self.use_quantum = False
                        distances[i, j] = classical_distance(X[i], self.centroids[j])
                else:
                    distances[i, j] = classical_distance(X[i], self.centroids[j])
        
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """Update centroids based on mean of points in each cluster."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for i in range(self.n_clusters):
            if np.sum(labels == i) == 0:
                new_centroids[i] = self.centroids[i]
            else:
                new_centroids[i] = np.mean(X[labels == i], axis=0)
                
        return new_centroids
    
    def fit(self, X):
        """Fit the QKMeans model to the data."""
        self.centroids = self.initialize_centroids(X)
        
        prev_centroids = np.zeros_like(self.centroids)
        
        for iteration in range(self.max_iter):
            prev_centroids = self.centroids.copy()
            
            self.labels_ = self.assign_clusters(X)
            
            self.centroids = self.update_centroids(X, self.labels_)
            
            if np.sum((self.centroids - prev_centroids) ** 2) < self.tol:
                print(f"Converged after {iteration+1} iterations")
                break
                
        return self
    
    def predict(self, X):
        """Predict cluster labels for samples in X."""
        return self.assign_clusters(X)

try:
    print("Attempting QKMeans clustering with quantum distance...")
    qkmeans = QKMeans(n_clusters=len(class_names))
    qkmeans.fit(X_normalized)
    predicted_labels = qkmeans.labels_
    
    ari_score = adjusted_rand_score(y, predicted_labels)
    silhouette =  silhouette_score(X_normalized, predicted_labels)
    
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i in range(len(class_names)):
        plt.scatter(X_normalized[y == i, 0], X_normalized[y == i, 1], label=f"Class {class_names[i]}")
    plt.scatter(qkmeans.centroids[:, 0], qkmeans.centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
    plt.title('True Classes')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    
    # Plot 2: Predicted clusters
    plt.subplot(1, 2, 2)
    for i in range(len(class_names)):
        plt.scatter(X_normalized[predicted_labels == i, 0], X_normalized[predicted_labels == i, 1], label=f'Cluster {i}')
    plt.scatter(qkmeans.centroids[:, 0], qkmeans.centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
    plt.title('QKMeans Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('qkmeans_wine_clustering.png', dpi=300)
    plt.show()
    
    # Create a confusion matrix-like table to visualize cluster-class relationships
    cluster_class_counts = np.zeros((len(class_names), len(class_names)))
    for i in range(len(y)):
        cluster_class_counts[predicted_labels[i], y[i]] += 1
    
    # Convert to a dataframe for better visualization
    df_confusion = pd.DataFrame(cluster_class_counts)
    df_confusion.columns = [f'Class {name}' for name in class_names]
    df_confusion.index = [f'Cluster {i}' for i in range(len(class_names))]
    print("\nCluster-Class Distribution:")
    print(df_confusion)
    
    # Calculate purity score (accuracy-like metric for clustering)
    purity = np.sum(np.max(cluster_class_counts, axis=1)) / np.sum(cluster_class_counts)
    print(f"\nPurity Score (Clustering Accuracy): {purity:.4f}")

except Exception as e:
    print(f"Error in quantum implementation: {e}")
    print("\nFalling back to classical KMeans implementation with quantum-inspired structure")
    
    # Classical fallback with similar structure
    qkmeans = QKMeans(n_clusters=len(class_names), use_quantum=False)
    qkmeans.fit(X_normalized)
    predicted_labels = qkmeans.labels_
    
    # Rest of the analysis continues as before...
    # Evaluate clustering performance
    ari_score = adjusted_rand_score(y, predicted_labels)
    silhouette = silhouette_score(X_normalized, predicted_labels)
    
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    
    # Visualize and output results as before...
    plt.figure(figsize=(12, 5))
    
    # Plot 1: True classes
    plt.subplot(1, 2, 1)
    for i in range(len(class_names)):
        plt.scatter(X_normalized[y == i, 0], X_normalized[y == i, 1], label=f"Class {class_names[i]}")
    plt.scatter(qkmeans.centroids[:, 0], qkmeans.centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
    plt.title('True Classes')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    
    # Plot 2: Predicted clusters
    plt.subplot(1, 2, 2)
    for i in range(len(class_names)):
        plt.scatter(X_normalized[predicted_labels == i, 0], X_normalized[predicted_labels == i, 1], label=f'Cluster {i}')
    plt.scatter(qkmeans.centroids[:, 0], qkmeans.centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
    plt.title('Classical KMeans Clusters (Quantum fallback)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('kmeans_fallback_clustering.png', dpi=300)
    plt.show()
    
    # Create a confusion matrix-like table
    cluster_class_counts = np.zeros((len(class_names), len(class_names)))
    for i in range(len(y)):
        cluster_class_counts[predicted_labels[i], y[i]] += 1
    
    # Convert to a dataframe for better visualization
    df_confusion = pd.DataFrame(cluster_class_counts)
    df_confusion.columns = [f'Class {name}' for name in class_names]
    df_confusion.index = [f'Cluster {i}' for i in range(len(class_names))]
    print("\nCluster-Class Distribution:")
    print(df_confusion)
    
    # Calculate purity score
    purity = np.sum(np.max(cluster_class_counts, axis=1)) / np.sum(cluster_class_counts)
    print(f"\nPurity Score (Clustering Accuracy): {purity:.4f}")
