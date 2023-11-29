import torch
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import logging
from torch_geometric.nn import global_mean_pool as pool_op

logger = logging.getLogger(__name__)

class GCN_regressor(torch.nn.Module):
    def __init__(self, atom_feature_dim, conv_hidden_channels, output_dim, regr_hidden_channels_1=128, regr_hidden_channels_2=64, regr_hidden_channels_3=32):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(atom_feature_dim, conv_hidden_channels)
        self.conv2 = GCNConv(conv_hidden_channels, conv_hidden_channels)
        self.conv3 = GCNConv(conv_hidden_channels, atom_feature_dim)
        self.fc1 = nn.Linear(atom_feature_dim + 1, regr_hidden_channels_1)
        self.fc2 = nn.Linear(regr_hidden_channels_1, regr_hidden_channels_2)
        self.fc3 = nn.Linear(regr_hidden_channels_2, regr_hidden_channels_3)
        self.fc4 = nn.Linear(regr_hidden_channels_3, output_dim)
        
    def forward(self, data):
        # Message passing
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.tanh(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.tanh(x)
        x = self.conv3(x, edge_index)

        # Readout and global pooling
        global_features = torch.cat([data.reactant_natoms.unsqueeze(1), data.reactant_nbonds.unsqueeze(1), data.reactant_mw.unsqueeze(1)], dim=1)
        x_global = global_features.mean(dim=1)

        # Repeat the global features for each node in the graph
        x_global = torch.index_select(x_global, 0, batch)
        x_global = x_global.unsqueeze(1)
        
        # Concatenate global features with node-level features
        x = torch.cat([x, x_global], dim=1)
        
        #readout
        x = pool_op(x, data.batch)

        # Property Prediction MLP
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x