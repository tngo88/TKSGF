# Construct graphs based on Top-K attribute similarity
# Import libraries
#region
import pandas as pd
import numpy as np
import torch
import faiss
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from datetime import datetime
from pathlib import Path
import math
import warnings
#endregion
warnings.filterwarnings('ignore')

#%% Define global parameters, load the dataset and preprocess the dataset
# Global parameters
datasetName = 'NF-BoT-IoT.csv'
train_percentage = 70
sim_threshold = 0.8
# Other parameters
dataset_path = ".../dataset/" + datasetName # Replace with the actual path
graph_dir = ".../sim_graph/tmp/" # Location of stored simulation graphs
# graph_dir = ".../sim_graph/test_Similarity/" + str(sim_threshold).replace(".", "") + "/"

df = pd.read_csv(dataset_path)

# Preprocess the dataset 1379275 / 600101
features = df.iloc[1:, 5:13].values  # Extract feature columns
labels = df.iloc[1:, -2].values  # Extract label column
labels_multiclass = df.iloc[1:, -1].values

# Encode categorical labels numerically
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Encode labels
le = LabelEncoder()
labels_multiclass = le.fit_transform(labels_multiclass)
labels = le.fit_transform(labels)

# Convert labels to a PyTorch tensor -> Recommended
labels = torch.tensor(labels, dtype=torch.long)
labels_multiclass = torch.tensor(labels_multiclass, dtype=torch.long)

#%% Split dataset into train, valuation and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=(100-train_percentage)/100, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
Xm_train, Xm_temp, ym_train, ym_temp = train_test_split(features, labels_multiclass, test_size=(100-train_percentage)/100, random_state=42)
Xm_val, Xm_test, ym_val, ym_test = train_test_split(Xm_temp, ym_temp, test_size=0.5, random_state=42)

#%% Create graph data with FAISS
def create_graph_data_with_faiss(features, labels, k = 10, type1='', type2=''):
    # Convert features to float32, which is required by FAISS -> for optimal performance
    float_features = features.astype('float32')

    # Normalize the feature vectors
    norm_features = float_features / np.linalg.norm(float_features, axis=1, keepdims=True)

    # Initialize FAISS index for cosine similarity (L2 distance equivalent)
    index = faiss.IndexFlatIP(norm_features.shape[1])
    # index = faiss.IndexFlatL2(float_features.shape[1]) # Initialize FAISS index for L2 (Euclidean) distance

    # Add features to the FAISS index
    index.add(norm_features)
    # index.add(float_features)

    # Search for similar vectors
    D, I = index.search(norm_features, k)  # Retrieve nearest neighbors
    # D, I = index.search(float_features, k)
    # D, I = index.search(norm_features, len(norm_features))  # Retrieve nearest neighbors

    # Create edges for nodes with a similarity distance below the threshold
    edge_index_undirected = []
    edge_index_directed = []
    for i in range(len(I)):
        for j, sim in zip(I[i], D[i]):
            if i != j and sim >= sim_threshold:  # Ensure no self-loops and meet the threshold
            # if i != j:
                edge_index_directed.append([i, j])  # Shape [num_edges, 2]
                if i < j:
                    edge_index_undirected.append([i, j])  # Added on 14 Feb 2025

    edge_index_undirected = torch.tensor(edge_index_undirected, dtype=torch.long).t().contiguous()  # Shape [2, num_edges]
    edge_index_directed = torch.tensor(edge_index_directed, dtype=torch.long).t().contiguous()  # Shape [2, num_edges]

    # Convert features and labels to PyTorch tensors -> Required as tested on 3 Feb 2025
    x = torch.tensor(norm_features, dtype=torch.float)
    # x = torch.tensor(float_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    # x = norm_features # Error if use it
    # y = labels # Error if use it

    sim_graph_undirected = Data(x=x, edge_index=edge_index_undirected, y=y)
    sim_graph_directed = Data(x=x, edge_index=edge_index_directed, y=y)

    if type2 == 'train':
        type2 = type2 + str(train_percentage)
    else:
        type2 = type2 + str(math.trunc((100 - train_percentage) / 2))

    graph_dir_undirected = (graph_dir + datasetName.split('.')[0] + '_' + type1 + '_undirected_' + type2 + '_k' + str(k) + ".pt")
    torch.save(sim_graph_undirected, graph_dir_undirected)
    graph_dir_directed = (graph_dir + datasetName.split('.')[0] + '_' + type1 + '_directed_' + type2 + '_k' + str(k) + ".pt")
    torch.save(sim_graph_directed, graph_dir_directed)

start_time_all = datetime.now()
ks = [3, 5, 7, 9, 10]  # Different values of k to try
for k in ks:
    start_time = datetime.now()
    create_graph_data_with_faiss(X_train, y_train, k, type1 = 'binary', type2 = 'train')  # Data(x=[80, 8], edge_index=[2, 588], y=[80])
    create_graph_data_with_faiss(X_val, y_val, k, type1 = 'binary', type2 = 'val')
    create_graph_data_with_faiss(X_test, y_test, k, type1 = 'binary', type2 = 'test')

    create_graph_data_with_faiss(Xm_train, ym_train, k, type1 = 'multi', type2 = 'train')  # Data(x=[80, 8], edge_index=[2, 588], y=[80])
    create_graph_data_with_faiss(Xm_val, ym_val, k, type1 = 'multi', type2 = 'val')
    create_graph_data_with_faiss(Xm_test, ym_test, k, type1 = 'multi', type2 = 'test')
    end_time = datetime.now()
    duration = end_time - start_time
    print(Path(dataset_path).stem + ' k = ' + str(k) + f" Duration time: {duration}")

end_time_all = datetime.now()
duration_all = end_time_all - start_time_all
print(Path(dataset_path).stem + f" Total duration time: {duration_all}")
