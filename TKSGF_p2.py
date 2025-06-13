# Use graph data generated from TKSGF_p1.py to generate classification report
# Import libraries
# region
import pandas as pd
import numpy as np
import torch
import faiss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler, NeighborLoader, ClusterData, ClusterLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm, GINConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch_optimizer import Lookahead
from torch.nn import Linear, LayerNorm, BatchNorm1d, Sequential, ReLU

from datetime import datetime
from itertools import product
from pathlib import Path
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import seaborn as sn
import matplotlib.pyplot as plt
import copy
import math
import csv
import warnings

# endregion
warnings.filterwarnings('ignore')

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # BatchNorm for stability
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # First GCN layer
        x = self.bn1(x)  # Normalize activations
        x = F.relu(x)  # Activation function after BatchNorm
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        x = self.conv2(x, edge_index)  # Second GCN layer
        return F.log_softmax(x, dim=1)


class GCN_1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN_1, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # First GCN layer
        x = F.relu(x)  # Activation function
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        x = self.conv2(x, edge_index)  # Second GCN layer
        return F.log_softmax(x, dim=1)


class GCN_2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN_2, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        x = self.conv1(x, edge_index)  # First GCN layer
        x = F.relu(x)  # Activation function
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        x = self.conv2(x, edge_index)  # Second GCN layer
        return F.log_softmax(x, dim=1)


class GCN1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # BatchNorm for stability
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # First GCN layer
        x = self.bn1(x)  # Normalize BEFORE activation
        x = F.relu(x)  # Activation function after BatchNorm
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GCN layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GCN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # BatchNorm for stability
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

        # Projection layer for residual connection (only if in_channels ≠ hidden_channels)
        self.residual_proj = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_residual = x  # Store original features

        # Apply projection if needed to match dimensions
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x_residual)

        x = self.conv1(x, edge_index)  # First GCN layer
        x = self.bn1(x)  # Apply BatchNorm BEFORE residual connection
        x = x + x_residual  # Add residual AFTER normalization
        x = F.relu(x)  # Activation function after BatchNorm
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GCN layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GCN3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN3, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # BatchNorm for stability
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

        # Projection layer for residual connection (only if in_channels ≠ hidden_channels)
        self.residual_proj = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_residual = x  # Store original features

        # Apply projection if needed to match dimensions
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x_residual)

        x = self.conv1(x, edge_index)  # First GCN layer
        x = self.bn1(x)  # Apply BatchNorm BEFORE residual connection
        x = F.relu(x)  # Activation function after BatchNorm
        x = x + x_residual  # Add residual
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GCN layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GCN4(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN4, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # BatchNorm for stability
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

        # Projection layer for residual connection (only if in_channels ≠ hidden_channels)
        self.residual_proj = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_residual = x  # Store original features

        # Apply projection if needed to match dimensions
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x_residual)

        x = self.conv1(x, edge_index)  # First GCN layer
        x = self.bn1(x + x_residual)  # BatchNorm AFTER residual connection
        x = F.relu(x)  # Activation function after BatchNorm
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GCN layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GCN5(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN5, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # BatchNorm for stability
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # First GCN layer
        x = F.relu(x)  # Activation function
        x = self.bn1(x)  # Normalize AFTER activation
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GCN layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GCN6(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN6, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # BatchNorm for stability
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

        # Projection layer for residual connection (only if in_channels ≠ hidden_channels)
        self.residual_proj = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_residual = x  # Store original features

        # Apply projection if needed to match dimensions
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x_residual)

        x = self.conv1(x, edge_index)  # First GCN layer
        x = F.relu(x)  # Activation function before BatchNorm
        x = self.bn1(x)  # Apply BatchNorm BEFORE residual connection
        x = x + x_residual  # Add residual
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GCN layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GCN7(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN7, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # BatchNorm for stability
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

        # Projection layer for residual connection (only if in_channels ≠ hidden_channels)
        self.residual_proj = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_residual = x  # Store original features

        # Apply projection if needed to match dimensions
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x_residual)

        x = self.conv1(x, edge_index)  # First GCN layer
        x = F.relu(x)  # Activation function before BatchNorm and residual connection
        x = self.bn1(x + x_residual)  # BatchNorm AFTER residual connection
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GCN layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GCN8(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN8, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # BatchNorm for stability
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

        # Projection layer for residual connection (only if in_channels ≠ hidden_channels)
        self.residual_proj = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_residual = x  # Store original features

        # Apply projection if needed to match dimensions
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x_residual)

        x = self.conv1(x, edge_index)  # First GCN layer
        x = x + x_residual  # Add residual
        x = F.relu(x)  # Activation function before BatchNorm and after residual connection
        x = self.bn1(x)  # BatchNorm AFTER residual connection
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GCN layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


# GAT Model Definition
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
        self.bn1 = BatchNorm1d(hidden_channels * 4)  # BatchNorm for stability
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # First GAT layer
        x = self.bn1(x)  # Normalize BEFORE activation
        x = F.elu(x)  # Activation function
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GAT layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities

class GAT_0(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GAT_0, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
        self.bn1 = BatchNorm1d(hidden_channels * 4)  # BatchNorm for stability
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False, dropout=dropout)
        self.bn2 = BatchNorm1d(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)  # Input dropout
        x = self.conv1(x, edge_index)  # First GAT layer
        x = self.bn1(x)  # Normalize BEFORE activation
        x = F.elu(x)  # Activation function
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GAT layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities

class GAT_1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GAT_1, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # First GAT layer
        x = F.elu(x)  # Activation function
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GAT layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GAT_2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GAT_2, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)  # Input dropout
        x = self.conv1(x, edge_index)  # First GAT layer
        x = F.elu(x)  # Activation function
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GAT layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GAT1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GAT1, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
        self.bn1 = BatchNorm1d(hidden_channels * 4)  # BatchNorm for stability
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False, dropout=dropout)
        self.bn2 = BatchNorm1d(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=self.dropout, training=self.training)  # Input dropout
        x = self.conv1(x, edge_index)  # First GAT layer
        x = self.bn1(x)  # Normalize BEFORE activation
        x = F.elu(x)  # Activation function after BatchNorm
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GAT layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GAT5(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GAT5, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
        self.bn1 = BatchNorm1d(hidden_channels * 4)  # BatchNorm for stability
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False, dropout=dropout)
        self.bn2 = BatchNorm1d(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # First GAT layer
        x = F.elu(x)  # Activation function
        x = self.bn1(x)  # Normalize AFTER activation
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GAT layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)  # Add BatchNorm
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # First GraphSAGE layer
        x = self.bn1(x)  # Normalize activations
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GraphSAGE layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GraphSAGE_1(torch.nn.Module):  # Equivalent to GraphSAGE9
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE_1, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=self.dropout, training=self.training)  # Input dropout
        x = self.conv1(x, edge_index)  # First GraphSAGE layer
        x = F.relu(x)  # Activation function
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GraphSAGE layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GraphSAGE_2(torch.nn.Module):  # Equivalent to GraphSAGE10
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE_2, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)  # Input dropout
        x = self.conv1(x, edge_index)  # First GraphSAGE layer
        x = F.relu(x)  # Activation function before BatchNorm
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GraphSAGE layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GraphSAGE1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE1, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)  # BatchNorm for stability
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm1d(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=self.dropout, training=self.training)  # Input dropout
        x = self.conv1(x, edge_index)  # First GraphSAGE layer
        x = self.bn1(x)  # Normalize BEFORE activation
        x = F.relu(x)  # Activation function after BatchNorm
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GraphSAGE layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GraphSAGE2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE2, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)  # BatchNorm for stability
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm1d(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

        # Projection layer for residual connection (only if in_channels ≠ hidden_channels)
        self.residual_proj = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_residual = x  # Store original features

        # Apply projection if needed to match dimensions
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x_residual)

        # x = F.dropout(x, p=self.dropout, training=self.training)  # Input dropout
        x = self.conv1(x, edge_index)  # First GraphSAGE layer
        x = self.bn1(x)  # Apply BatchNorm BEFORE residual connection
        x = x + x_residual  # Add residual AFTER normalization
        # x = self.bn1(x + x_residual)  # BatchNorm AFTER residual connection
        x = F.relu(x)  # Activation function after BatchNorm
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GraphSAGE layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GraphSAGE3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE3, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)  # BatchNorm for stability
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm1d(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

        # Projection layer for residual connection (only if in_channels ≠ hidden_channels)
        self.residual_proj = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_residual = x  # Store original features

        # Apply projection if needed to match dimensions
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x_residual)

        # x = F.dropout(x, p=self.dropout, training=self.training)  # Input dropout
        x = self.conv1(x, edge_index)  # First GraphSAGE layer
        x = self.bn1(x)  # Apply BatchNorm BEFORE residual connection
        x = F.relu(x)  # Activation function after BatchNorm
        x = x + x_residual  # Add residual
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GraphSAGE layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GraphSAGE4(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE4, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)  # BatchNorm for stability
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm1d(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

        # Projection layer for residual connection (only if in_channels ≠ hidden_channels)
        self.residual_proj = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_residual = x  # Store original features

        # Apply projection if needed to match dimensions
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x_residual)

        # x = F.dropout(x, p=self.dropout, training=self.training)  # Input dropout
        x = self.conv1(x, edge_index)  # First GraphSAGE layer
        # x = self.bn1(x)  # Apply BatchNorm BEFORE residual connection
        # x = x + x_residual  # Add residual AFTER normalization
        x = self.bn1(x + x_residual)  # BatchNorm AFTER residual connection
        x = F.relu(x)  # Activation function after BatchNorm
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GraphSAGE layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GraphSAGE5(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE5, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)  # BatchNorm for stability
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm1d(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # First GraphSAGE layer
        x = F.relu(x)  # Activation function
        x = self.bn1(x)  # Normalize AFTER activation
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GraphSAGE layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GraphSAGE6(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE6, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)  # BatchNorm for stability
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm1d(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

        # Projection layer for residual connection (only if in_channels ≠ hidden_channels)
        self.residual_proj = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_residual = x  # Store original features

        # Apply projection if needed to match dimensions
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x_residual)

        # x = F.dropout(x, p=self.dropout, training=self.training)  # Input dropout
        x = self.conv1(x, edge_index)  # First GraphSAGE layer
        x = F.relu(x)  # Activation function before BatchNorm
        x = self.bn1(x)  # Apply BatchNorm BEFORE residual connection
        x = x + x_residual  # Add residual
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GraphSAGE layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GraphSAGE7(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE7, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)  # BatchNorm for stability
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm1d(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

        # Projection layer for residual connection (only if in_channels ≠ hidden_channels)
        self.residual_proj = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_residual = x  # Store original features

        # Apply projection if needed to match dimensions
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x_residual)

        # x = F.dropout(x, p=self.dropout, training=self.training)  # Input dropout
        x = self.conv1(x, edge_index)  # First GraphSAGE layer
        x = F.relu(x)  # Activation function before BatchNorm and residual connection
        x = self.bn1(x + x_residual)  # BatchNorm AFTER residual connection
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GraphSAGE layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities


class GraphSAGE8(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE8, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)  # BatchNorm for stability
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.bn2 = BatchNorm1d(out_channels)  # BatchNorm for second layer
        self.dropout = dropout  # Store dropout rate

        # Projection layer for residual connection (only if in_channels ≠ hidden_channels)
        self.residual_proj = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_residual = x  # Store original features

        # Apply projection if needed to match dimensions
        if self.residual_proj is not None:
            x_residual = self.residual_proj(x_residual)

        # x = F.dropout(x, p=self.dropout, training=self.training)  # Input dropout
        x = self.conv1(x, edge_index)  # First GraphSAGE layer
        x = x + x_residual  # Add residual
        x = F.relu(x)  # Activation function before BatchNorm and after residual connection
        x = self.bn1(x)  # BatchNorm AFTER residual connection
        x = F.dropout(x, p=self.dropout, training=self.training)  # Hidden layer dropout
        x = self.conv2(x, edge_index)  # Second GraphSAGE layer
        x = self.bn2(x)  # Normalize second layer
        return F.log_softmax(x, dim=1)  # Always return log probabilities

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GIN, self).__init__()

        # First GIN layer with MLP
        nn1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm1d(hidden_channels)  # Add BatchNorm

        # Second GIN layer with MLP
        nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
        self.conv2 = GINConv(nn2)

        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # First GIN layer
        x = self.bn1(x)  # Normalize activations
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # Always return log probabilities

class GraphSAINT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAINT, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout  # Store dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Training function
def train(sim_graph):
    model.train()
    # for data in train_loader:  # Not required as train_graph is a single graph
    optimizer.zero_grad()  # Clear the gradients from the previous step
    data = sim_graph.to(device)  # Added on 3 Feb 2025
    out = model(data)  # Forward pass: compute predictions
    loss = loss_fn(out, data.y)  # Compute the loss (Output, target)
    loss.backward()  # Backward pass: compute gradients
    optimizer.step()  # Update model parameters based on gradients
    return loss.item()

def train_GraphSAINT(loader):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()  # Clear the gradients from the previous step
        data = batch.to(device)  # Added on 3 Feb 2025
        out = model(data)  # Forward pass: compute predictions
        loss = loss_fn(out, data.y)  # Compute the loss (Output, target)
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters based on gradients
        total_loss += loss.item()
    return total_loss


# Evaluation function #No change
def evaluate(sim_graph):
    model.eval()  # Set to evaluation mode. Disables dropout and batch normalization layers from updating their statistics.
    with torch.no_grad():  # Disable gradient computation
        # for data in loader:  # Not required as test_graph is a single graph
        data = sim_graph.to(device)  # Added on 5 Feb 2025
        out = model(data)
        loss = loss_fn(out, data.y)  # Compute the loss (Output, target)

        pred = out.argmax(dim=1)  # selects the index of the highest value for each row
        correct = (pred == data.y).sum().item()
        acc = correct / data.y.size(0)

        ground_labels = data.y.cpu().numpy()
        predicted_labels = pred.cpu().numpy()
        f1 = f1_score(ground_labels, predicted_labels, average='weighted')  # Adjust 'average' as needed
    return ground_labels, predicted_labels, loss.item(), acc, f1

def evaluate_GraphSAINT(loader):
    model.eval()  # Set to evaluation mode. Disables dropout and batch normalization layers from updating their statistics.
    total_loss = 0
    total_correct = 0
    total_data = 0
    ground_labels = np.empty((0,))
    predicted_labels = np.empty((0,))
    count = 0
    with torch.no_grad():  # Disable gradient computation
        for data in loader:  # Not required as test_graph is a single graph
            # count = count + 1
            data = data.to(device)  # Added on 5 Feb 2025
            out = model(data)
            loss = loss_fn(out, data.y)  # Compute the loss (Output, target)
            total_loss += loss.item()

            pred = out.argmax(dim=1)  # selects the index of the highest value for each row
            correct = (pred == data.y).sum().item()
            total_correct += correct
            total_data += data.y.size(0)

            if np.array(ground_labels).size == 0:
                ground_labels = data.y.cpu().numpy()
            else:
                ground_labels = np.concatenate((ground_labels, data.y.cpu().numpy()))
            if np.array(predicted_labels).size == 0:
                predicted_labels = pred.cpu().numpy()
            else:
                predicted_labels = np.concatenate((predicted_labels, pred.cpu().numpy()))
            # if count%100 == 0:
            #     print(batch)
            #     print('count = ' + str(count))
            #     print(ground_labels)
            #     print(predicted_labels)


        acc = total_correct / total_data
        f1 = f1_score(ground_labels, predicted_labels, average='weighted')  # Adjust 'average' as needed
    return ground_labels, predicted_labels, total_loss, acc, f1


# Function to find the best model through early stopping
def get_best_model():
    min_delta = 0
    trigger_times = 0  # Early stopping trigger indicator
    trigger = False
    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    patience = 4
    for epoch in range(epochs):
        train_loss = train(train_graph)
        # accuracy = evaluate(test_loader)  # Not required as test_graph is a single graph
        _, _, val_loss, val_acc, val_f1 = evaluate(val_graph)
        # print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},"
        #       f" Val Acc: {val_acc * 100:.4f}%, Val F1: {val_f1 * 100:.4f}%")
        if best_val_loss - val_loss > min_delta:
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            trigger_times = 0
            trigger = False  # Added on 24 Feb 2025
            # print('Current best model is at epoch = ' + str(best_epoch) + ', cur_patience = ' + str(patience))
        else:
            trigger_times += 1
            if not trigger:  # and abs(val_loss - best_val_loss) < min_delta:  # Updated on 24 Feb 2025
                min_delta = abs(val_loss - best_val_loss)
                patience = min(patience + 1, max_patience)  # Increase patience
                trigger = True
            # print('trigger_times = ' + str(trigger_times) + ', min_delta = ' + str(min_delta) + ', cur_patience = ' + str(
            #     patience))
            if trigger_times >= patience and epoch+1 >= max_patience:
                print(f'Early stopping triggered after {epoch+1} epochs' + ', cur_patience = ' + str(
                    patience) + ', max_patience = ' + str(
                    max_patience) + ', best_epoch = ' + str(best_epoch))
                break
        if epoch == epochs - 1:
            print(f'No early stopping triggered after {epochs} epochs')
            continue
    return patience, best_model, best_epoch

def get_best_model_GraphSAINT(train_loader, val_loader):
    min_delta = 0
    trigger_times = 0  # Early stopping trigger indicator
    trigger = False
    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    patience = 4
    for epoch in range(epochs):
        train_loss = train_GraphSAINT(train_loader)
        # accuracy = evaluate(test_loader)  # Not required as test_graph is a single graph
        _, _, val_loss, val_acc, val_f1 = evaluate_GraphSAINT(val_loader)
        # print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},"
        #       f" Val Acc: {val_acc * 100:.4f}%, Val F1: {val_f1 * 100:.4f}%")
        if best_val_loss - val_loss > min_delta:
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            trigger_times = 0
            trigger = False  # Added on 24 Feb 2025
            # print('Current best model is at epoch = ' + str(best_epoch) + ', cur_patience = ' + str(patience))
        else:
            trigger_times += 1
            if not trigger:  # and abs(val_loss - best_val_loss) < min_delta:  # Updated on 24 Feb 2025
                min_delta = abs(val_loss - best_val_loss)
                patience = min(patience + 1, max_patience)  # Increase patience
                trigger = True
            # print('trigger_times = ' + str(trigger_times) + ', min_delta = ' + str(min_delta) + ', cur_patience = ' + str(
            #     patience))
            if trigger_times >= patience and epoch+1 >= max_patience:
                print(f'Early stopping triggered after {epoch+1} epochs' + ', cur_patience = ' + str(
                    patience) + ', max_patience = ' + str(
                    max_patience) + ', best_epoch = ' + str(best_epoch))
                break
        if epoch == epochs - 1:
            print(f'No early stopping triggered after {epochs} epochs')
            continue
    return patience, best_model, best_epoch

# %% Define global parameters, load the dataset and preprocess the dataset
# Global parameters
# datasetNames = ['NF-BoT-IoT.csv', 'NF-ToN-IoT.csv']
# classify_types = ['binary_undirected', 'binary_directed', 'multi_undirected', 'multi_directed']
# datasetNames = ['NF-ToN-IoT.csv']
# classify_types = ['multi_directed']
# approach = "Cosine + Sim09 + TopK"
# sim_threshold = 0.9 #1111
# ks = [7]

learning_rate = 0.01  # 0.01, 0.005
hidden_channels = 32
epochs = 2000

# Other parameters
# dataset_path = ".../dataset/" + datasetName  # Replace with the actual path
overleaf_dir = ".../Overleaf/tmp/"
graph_dir = ".../sim_graph/"  # Location to save simulation graphs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# [System.Environment]::SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0", "User")
# nvidia-smi
torch.set_float32_matmul_precision('high')  # Speeds up computations

# gpu_count = torch.cuda.device_count()
# print(f"Number of GPUs: {gpu_count}")
# for i in range(gpu_count):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

start_time_all = datetime.now()
header_BoT_binary = False
header_ToN_binary = False
header_BoT_multi = False
header_ToN_multi = False
# ii = [0,1,2,4]
ii = [4]
sim_thresholds = [0.8]
test_type = "Normal"  #test_Euclidean, test_noSimilarity, test_Similarity
# for datasetName, classify_type, k in product(datasetNames, classify_types, ks):
for ii, sim_threshold in product(ii, sim_thresholds):  #11111
    if ii == 0:
        datasetName = 'NF-BoT-IoT.csv'
        classify_type = "binary_directed"
        k = 9
    elif ii == 1:
        datasetName = 'NF-BoT-IoT.csv'
        classify_type = "multi_undirected"
        k = 10
    elif ii == 2:
        datasetName = 'NF-ToN-IoT.csv'
        classify_type = "binary_directed"
        k = 7
    elif ii == 3:
        datasetName = 'NF-ToN-IoT.csv'
        classify_type = "multi_directed"
        k = 7
    elif ii == 4:
        datasetName = 'NF-ToN-IoT.csv'
        classify_type = "multi_undirected"
        k = 5

    approach = "Cosine + Sim" + str(sim_threshold).replace(".", "") + " + TopK"
    if test_type == "test_Similarity":
        graph_dir = "C:/Users/tdngo/OneDrive/Desktop/4664318/PhD/PyCharm/myGNN/pythonProject/sim_graph/test_Similarity/" + str(sim_threshold).replace(".", "") + "/"
    if test_type == "test_Euclidean":
        approach = "Euclidean + Sim" + str(sim_threshold).replace(".", "") + " + TopK"
        graph_dir = "C:/Users/tdngo/OneDrive/Desktop/4664318/PhD/PyCharm/myGNN/pythonProject/sim_graph/test_Euclidean/"
    elif test_type == "test_noSimilarity":
        approach = "Cosine + TopK"
        graph_dir = "C:/Users/tdngo/OneDrive/Desktop/4664318/PhD/PyCharm/myGNN/pythonProject/sim_graph/test_noSimilarity/"
    start_time = datetime.now()
    max_patience = 10 if datasetName == 'NF-BoT-IoT.csv' else 20
    train_percentage = 70 if datasetName == 'NF-BoT-IoT.csv' else 60
    result_path = "C:/Users/tdngo/OneDrive/Desktop/4664318/PhD/PyCharm/myGNN/pythonProject/results/" + \
                  datasetName.split('.')[0]
    if 'binary' in classify_type:
        result_path = result_path + "_GNN_binary_test_results.csv"
        result_title = ["Dataset", "Approach", "train%", "classify_type", "k", "GNN_model", "hidden_channel",
                        "learning_rate",
                        "patience", "best_epoch", "epoch", "test_confusion_matrix_label",
                        "test_acc_label", "test_pre_label", "test_rec_label", "test_f1_label",
                        "test_class0_pre_label", "test_class0_rec_label", "test_class0_f1_label",
                        "test_class1_pre_label", "test_class1_rec_label", "test_class1_f1_label",
                        "Device", "Duration"
                        ]
        if datasetName == 'NF-BoT-IoT.csv' and header_BoT_binary == False:
            with open(result_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result_title)
            header_BoT_binary = True
        elif datasetName == 'NF-ToN-IoT.csv' and header_ToN_binary == False:
            with open(result_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result_title)
            header_ToN_binary = True
    elif 'multi' in classify_type:
        result_path = result_path + "_GNN_multi_test_results.csv"
        if datasetName == 'NF-BoT-IoT.csv' and header_BoT_multi == False:
            result_title = ["Dataset", "Approach", "train%", "classify_type", "k", "GNN_model", "hidden_channel",
                            "learning_rate",
                            "patience", "best_epoch", "epoch", "test_confusion_matrix_label",
                            "test_acc_label", "test_pre_label", "test_rec_label", "test_f1_label",
                            "test_Benign_pre", "test_Benign_rec", "test_Benign_f1",
                            "test_DDoS_pre", "test_DDoS_rec", "test_DDoS_f1",
                            "test_DoS_pre", "test_DoS_rec", "test_DoS_f1",
                            "test_Reconnaissance_pre", "test_Reconnaissance_rec", "test_Reconnaissance_f1",
                            "test_Theft_pre", "test_Theft_rec", "test_Theft_f1",
                            "Device", "Duration"
                            ]
            with open(result_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result_title)
            header_BoT_multi = True
        elif datasetName == 'NF-ToN-IoT.csv' and header_ToN_multi == False:
            result_title = ["Dataset", "Approach", "train%", "classify_type", "k", "GNN_model", "hidden_channel",
                            "learning_rate",
                            "patience", "best_epoch", "epoch", "test_confusion_matrix_label",
                            "test_acc_label", "test_pre_label", "test_rec_label", "test_f1_label",
                            "test_Benign_pre", "test_Benign_rec", "test_Benign_f1",
                            "test_Backdoor_pre", "test_Backdoor_rec", "test_Backdoor_f1",
                            "test_DDoS_pre", "test_DDoS_rec", "test_DDoS_f1",
                            "test_DoS_pre", "test_DoS_rec", "test_DoS_f1",
                            "test_Injection_pre", "test_Injection_rec", "test_Injection_f1",
                            "test_MITM_pre", "test_MITM_rec", "test_MITM_f1",
                            "test_Password_pre", "test_Password_rec", "test_Password_f1",
                            "test_Ransomware_pre", "test_Ransomware_rec", "test_Ransomware_f1",
                            "test_Scanning_pre", "test_Scanning_rec", "test_Scanning_f1",
                            "test_XSS_pre", "test_XSS_rec", "test_XSS_f1",
                            "Device", "Duration"
                            ]
            with open(result_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result_title)
            header_ToN_multi = True

    # Load graph data
    train_graph_path = graph_dir + datasetName.split('.')[0] + '_' + classify_type + '_train' + str(
        train_percentage) + '_k' + str(k) + ".pt"
    val_graph_path = graph_dir + datasetName.split('.')[0] + '_' + classify_type + '_val' + str(
        math.trunc((100 - train_percentage) / 2)) + '_k' + str(k) + ".pt"
    test_graph_path = graph_dir + datasetName.split('.')[0] + '_' + classify_type + '_test' + str(
        math.trunc((100 - train_percentage) / 2)) + '_k' + str(k) + ".pt"
    train_graph = torch.load(train_graph_path, weights_only=False)
    val_graph = torch.load(val_graph_path, weights_only=False)
    test_graph = torch.load(test_graph_path, weights_only=False)
    num_features = train_graph.x.shape[1]
    num_classes = len(torch.unique(train_graph.y))

    # Initialize model, optimizer, and loss function
    myGCN = GCN(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(device)
    myGCN_1 = GCN_1(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGCN_2 = GCN_2(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGCN1 = GCN1(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGCN2 = GCN2(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGCN3 = GCN3(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGCN4 = GCN4(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGCN5 = GCN5(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGCN6 = GCN6(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGCN7 = GCN7(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGCN8 = GCN8(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGAT = GAT(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(device)
    myGAT_0 = GAT_0(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGAT_1 = GAT_1(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGAT_2 = GAT_2(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGAT1 = GAT1(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGAT5 = GAT5(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGraphSAGE = GraphSAGE(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGraphSAGE_1 = GraphSAGE_1(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGraphSAGE_2 = GraphSAGE_2(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGraphSAGE1 = GraphSAGE1(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGraphSAGE2 = GraphSAGE2(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGraphSAGE3 = GraphSAGE3(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGraphSAGE4 = GraphSAGE4(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGraphSAGE5 = GraphSAGE5(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGraphSAGE6 = GraphSAGE6(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGraphSAGE7 = GraphSAGE7(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGraphSAGE8 = GraphSAGE8(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(
        device)
    myGIN = GIN(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(device)
    myGraphSAINT = GraphSAINT(in_channels=num_features, hidden_channels=hidden_channels, out_channels=num_classes).to(device)

    # models = [myGCN, myGCN_1, myGCN_2, myGCN1, myGCN2, myGCN3, myGCN4, myGCN5, myGCN6, myGCN7, myGCN8,
    #           myGAT, myGAT_0, myGAT_1, myGAT_2, myGAT1, myGAT5,
    #           myGraphSAGE, myGraphSAGE_1, myGraphSAGE_2, myGraphSAGE1, myGraphSAGE2, myGraphSAGE3,
    #           myGraphSAGE4, myGraphSAGE5, myGraphSAGE6, myGraphSAGE7, myGraphSAGE8]
    # gnn_types = ["GCN", "GCN_1", "GCN_2", "GCN1", "GCN2", "GCN3", "GCN4", "GCN5", "GCN6", "GCN7", "GCN8",
    #              "GAT", "GAT_0", "GAT_1", "GAT_2", "GAT1", "GAT5",
    #              "GraphSAGE", "GraphSAGE_1", "GraphSAGE_2", "GraphSAGE1", "GraphSAGE2", "GraphSAGE3",
    #              "GraphSAGE4", "GraphSAGE5", "GraphSAGE6", "GraphSAGE7", "GraphSAGE8"]

    # models = [myGCN, myGCN_1, myGCN_2, myGCN1, myGCN2, myGCN3, myGCN4, myGCN5, myGCN6,  myGCN7, myGCN8]
    # gnn_types = ["GCN", "GCN_1", "GCN_2", "GCN1", "GCN2", "GCN3", "GCN4", "GCN5", "GCN6", "GCN7", "GCN8"]
    # models = [myGAT, myGAT_0, myGAT_1, myGAT_2, myGAT1, myGAT5]
    # gnn_types = ["GAT", "GAT_0", "GAT_1", "GAT_2", "GAT1", "GAT5"]
    # models = [myGraphSAGE, myGraphSAGE_1, myGraphSAGE_2, myGraphSAGE1, myGraphSAGE2, myGraphSAGE3,
    #           myGraphSAGE4, myGraphSAGE5, myGraphSAGE6, myGraphSAGE7, myGraphSAGE8]
    # gnn_types = ["GraphSAGE", "GraphSAGE_1", "GraphSAGE_2", "GraphSAGE1", "GraphSAGE2", "GraphSAGE3",
    #              "GraphSAGE4", "GraphSAGE5", "GraphSAGE6", "GraphSAGE7", "GraphSAGE8"]
    # models = [myGraphSAGE]
    # gnn_types = ["GraphSAGE"]

    if ii == 0:
        models = [myGraphSAGE]
        gnn_types = ["GraphSAGE"]
    elif ii == 1:
        models = [myGraphSAGE]
        gnn_types = ["GraphSAGE"]
    elif ii == 2:
        models = [myGraphSAGE_1]
        gnn_types = ["GraphSAGE_1"]
    else:
        models = [myGraphSAGE3]
        gnn_types = ["GraphSAGE3"]

    for i in range(len(models)):
        gnn_type = gnn_types[i]
        print('Running ' + datasetName + ', ' + approach + ', train% = ' + str(
            train_percentage) + ', ' + classify_type + ', k = ' + str(
            k) + ', ' + gnn_type + ', hidden_channels = ' + str(
            hidden_channels) + ', learning_rate = ' + str(learning_rate)
              + ', patience = adaptive')
        model = models[i]
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # No change
        base_optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)  # Sync every 5 steps
        loss_fn = torch.nn.NLLLoss()  # CrossEntropyLoss, NLLLoss

        # Train and evaluate the model
        # if gnn_type == "GraphSAINT": ### NOT WORKING
        #
        #     train_cluster_data = ClusterData(train_graph, num_parts=100, recursive=False)
        #     val_cluster_data = ClusterData(val_graph, num_parts=100, recursive=False)
        #     test_cluster_data = ClusterData(test_graph, num_parts=100, recursive=False)
        #
        #     train_loader = ClusterLoader(train_cluster_data, batch_size=10, shuffle=True)
        #     val_loader = ClusterLoader(val_cluster_data, batch_size=10, shuffle=True)
        #     test_loader = ClusterLoader(test_cluster_data, batch_size=10, shuffle=True)
        #
        #     # train_loader = DataLoader(train_graph, batch_size=32, shuffle=True)
        #     # val_loader = DataLoader(val_graph, batch_size=32, shuffle=True)
        #     # test_loader = DataLoader(test_graph, batch_size=32, shuffle=True)
        #     # train_loader = NeighborLoader(
        #     #     train_graph, num_neighbors=[15, 10],  # simulate graph sampling (2-hop neighborhood)
        #     #     batch_size=128,
        #     #     shuffle=True
        #     # )
        #     # val_loader = NeighborLoader(
        #     #     val_graph, num_neighbors=[15, 10],  # simulate graph sampling (2-hop neighborhood)
        #     #     batch_size=128,
        #     #     shuffle=False
        #     # )
        #     # test_loader = NeighborLoader(
        #     #     test_graph, num_neighbors=[15, 10],  # simulate graph sampling (2-hop neighborhood)
        #     #     batch_size=128,
        #     #     shuffle=False
        #     # )
        #     patience, model, final_epoch = get_best_model_GraphSAINT(train_loader, val_loader)
        #     val_true_labels, val_pred_labels, val_loss, val_acc, val_f1 = evaluate_GraphSAINT(val_graph)
        #     test_true_labels, test_pred_labels, test_loss, test_acc, test_f1 = evaluate_GraphSAINT(test_graph)
        # else:
        patience, model, final_epoch = get_best_model()
        val_true_labels, val_pred_labels, val_loss, val_acc, val_f1 = evaluate(val_graph)
        test_true_labels, test_pred_labels, test_loss, test_acc, test_f1 = evaluate(test_graph)

        # Final Test Accuracy
        test_confusion_matrix_label = confusion_matrix(test_true_labels, test_pred_labels)

        print(f'Val Loss: {val_loss}, Val Acc: {val_acc * 100:.4f}%, Val F1: {val_f1 * 100:.4f}%')
        print(f'Test Loss: {test_loss}, Test Acc: {test_acc * 100:.4f}%, Test F1: {test_f1 * 100:.4f}%')

        # Generate the classification report
        # predictions = model(test_graph).argmax(dim=1).to('cpu')
        # labels = test_graph.y.to('cpu')
        # print(torch.equal(predictions, y_test_pred))
        # print(torch.equal(labels, y_test))
        my_labels = np.unique(test_true_labels).tolist()
        my_target_names = ['class ' + str(ele) for ele in np.unique(test_true_labels).tolist()]
        test_report = classification_report(test_true_labels, test_pred_labels, labels=my_labels,
                                            target_names=my_target_names,
                                            digits=6, output_dict=True, zero_division=0)
        # report = classification_report(y_test, y_pred)
        # print(report)

        test_acc_label = test_report['accuracy']
        test_pre_label = test_report['weighted avg']['precision']
        test_rec_label = test_report['weighted avg']['recall']
        test_f1_label = test_report['weighted avg']['f1-score']
        test_class0_pre_label = test_report[my_target_names[0]]['precision']
        test_class0_rec_label = test_report[my_target_names[0]]['recall']
        test_class0_f1_label = test_report[my_target_names[0]]['f1-score']
        test_class1_pre_label = test_report[my_target_names[1]]['precision']
        test_class1_rec_label = test_report[my_target_names[1]]['recall']
        test_class1_f1_label = test_report[my_target_names[1]]['f1-score']

        if classify_type.split('_')[0] == 'multi':
            cma = confusion_matrix(test_true_labels, test_pred_labels)
            cman = cma.astype('float') / cma.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots(figsize=(10, 8))  # Larger figure for clarity
            sn.heatmap(cman, annot=True, fmt='.2f', annot_kws={"size": 10}, cmap='Blues', cbar=True)

            if datasetName == 'NF-BoT-IoT.csv':
                tick_labels = ['Benign', 'DDoS', 'DoS', 'Reconnaissance', 'Theft']
            elif datasetName == 'NF-ToN-IoT.csv':
                tick_labels = ['Benign', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'MITM', 'Password', 'Ransomware',
                               'Scanning', 'XSS']
            else:
                tick_labels = []  # Default fallback in case dataset name is not recognized

            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(tick_labels, rotation=0, fontsize=10)
            ax.set_xlabel("Predicted Attack Label", fontsize=12)
            ax.set_ylabel("True Attack Label", fontsize=12)
            # ax.set_title(f"Confusion Matrix - {datasetName.split('.')[0]}", fontsize=14)

            plt.tight_layout()
            plt.savefig(overleaf_dir + f'myConfusionMulti_{datasetName.split(".")[0]}.png', dpi=600)

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Duration time: {duration}")

        # save the results
        with open(result_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if 'binary' in classify_type:
                result_list = [datasetName.split('.')[0], approach, train_percentage, classify_type, k, gnn_type, hidden_channels,
                               learning_rate,
                               patience, final_epoch, epochs, test_confusion_matrix_label,
                               test_acc_label, test_pre_label, test_rec_label, test_f1_label,
                               test_class0_pre_label, test_class0_rec_label, test_class0_f1_label,
                               test_class1_pre_label, test_class1_rec_label, test_class1_f1_label,
                               device, duration
                               ]
            elif 'multi' in classify_type:
                test_class2_pre_label = test_report[my_target_names[2]]['precision']
                test_class2_rec_label = test_report[my_target_names[2]]['recall']
                test_class2_f1_label = test_report[my_target_names[2]]['f1-score']
                test_class3_pre_label = test_report[my_target_names[3]]['precision']
                test_class3_rec_label = test_report[my_target_names[3]]['recall']
                test_class3_f1_label = test_report[my_target_names[3]]['f1-score']
                test_class4_pre_label = test_report[my_target_names[4]]['precision']
                test_class4_rec_label = test_report[my_target_names[4]]['recall']
                test_class4_f1_label = test_report[my_target_names[4]]['f1-score']
                if datasetName == 'NF-BoT-IoT.csv':
                    result_list = [datasetName.split('.')[0], approach, train_percentage, classify_type, k, gnn_type,
                                   hidden_channels,
                                   learning_rate,
                                   patience, final_epoch, epochs, test_confusion_matrix_label,
                                   test_acc_label, test_pre_label, test_rec_label, test_f1_label,
                                   test_class0_pre_label, test_class0_rec_label, test_class0_f1_label,
                                   test_class1_pre_label, test_class1_rec_label, test_class1_f1_label,
                                   test_class2_pre_label, test_class2_rec_label, test_class2_f1_label,
                                   test_class3_pre_label, test_class3_rec_label, test_class3_f1_label,
                                   test_class4_pre_label, test_class3_rec_label, test_class4_f1_label,
                                   device, duration
                                   ]
                elif datasetName == 'NF-ToN-IoT.csv':
                    test_class5_pre_label = test_report[my_target_names[5]]['precision']
                    test_class5_rec_label = test_report[my_target_names[5]]['recall']
                    test_class5_f1_label = test_report[my_target_names[5]]['f1-score']
                    test_class6_pre_label = test_report[my_target_names[6]]['precision']
                    test_class6_rec_label = test_report[my_target_names[6]]['recall']
                    test_class6_f1_label = test_report[my_target_names[6]]['f1-score']
                    test_class7_pre_label = test_report[my_target_names[7]]['precision']
                    test_class7_rec_label = test_report[my_target_names[7]]['recall']
                    test_class7_f1_label = test_report[my_target_names[7]]['f1-score']
                    test_class8_pre_label = test_report[my_target_names[8]]['precision']
                    test_class8_rec_label = test_report[my_target_names[8]]['recall']
                    test_class8_f1_label = test_report[my_target_names[8]]['f1-score']
                    test_class9_pre_label = test_report[my_target_names[9]]['precision']
                    test_class9_rec_label = test_report[my_target_names[9]]['recall']
                    test_class9_f1_label = test_report[my_target_names[9]]['f1-score']
                    result_list = [datasetName.split('.')[0], approach, train_percentage, classify_type, k, gnn_type,
                                   hidden_channels,
                                   learning_rate,
                                   patience, final_epoch, epochs, test_confusion_matrix_label,
                                   test_acc_label, test_pre_label, test_rec_label, test_f1_label,
                                   test_class0_pre_label, test_class0_rec_label, test_class0_f1_label,
                                   test_class1_pre_label, test_class1_rec_label, test_class1_f1_label,
                                   test_class2_pre_label, test_class2_rec_label, test_class2_f1_label,
                                   test_class3_pre_label, test_class3_rec_label, test_class3_f1_label,
                                   test_class4_pre_label, test_class4_rec_label, test_class4_f1_label,
                                   test_class5_pre_label, test_class5_rec_label, test_class5_f1_label,
                                   test_class6_pre_label, test_class6_rec_label, test_class6_f1_label,
                                   test_class7_pre_label, test_class7_rec_label, test_class7_f1_label,
                                   test_class8_pre_label, test_class8_rec_label, test_class8_f1_label,
                                   test_class9_pre_label, test_class9_rec_label, test_class9_f1_label,
                                   device, duration
                                   ]
            writer.writerow(result_list)

end_time_all = datetime.now()
duration_all = end_time_all - start_time_all
print(f"Total duration time: {duration_all}")
