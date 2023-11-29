import torch
from torch_geometric.nn import GCNConv
from torch import nn
import logging
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as pool_op
from torch_scatter import scatter_mean

logger = logging.getLogger(__name__)

class GCN_regressor_w_bond(torch.nn.Module):
    def __init__(self, atom_feature_dim, bond_feature_dim, node_hidden_channels_1,
                 edge_hidden_channels_1, message_channels, 
                 regr_hidden_channels_1=128, regr_hidden_channels_2=64, regr_hidden_channels_3=32,
                 output_dim = 1):
        super(GCN_regressor_w_bond, self).__init__()
        torch.manual_seed(12345)
        self.node_conv1 = GCNConv(atom_feature_dim, node_hidden_channels_1)
        self.node_conv2 = GCNConv(node_hidden_channels_1, node_hidden_channels_1)
        self.node_conv3 = GCNConv(node_hidden_channels_1, atom_feature_dim)
        
        self.edge_conv1 = GCNConv(bond_feature_dim, edge_hidden_channels_1)
        self.edge_conv2 = GCNConv(edge_hidden_channels_1, edge_hidden_channels_1)
        self.edge_conv3 = GCNConv(edge_hidden_channels_1, bond_feature_dim)
        
        self.node_transform = nn.Linear(atom_feature_dim, message_channels)
        
        self.message_transform = nn.Linear(message_channels + bond_feature_dim, atom_feature_dim)
        
        self.fc1 = nn.Linear(atom_feature_dim + 1, regr_hidden_channels_1)
        self.fc2 = nn.Linear(regr_hidden_channels_1, regr_hidden_channels_2)
        self.fc3 = nn.Linear(regr_hidden_channels_2, regr_hidden_channels_3)
        self.fc4 = nn.Linear(regr_hidden_channels_3, output_dim)
        
    def forward(self, data):
        # Message passing
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        #Node Feature update
    
        x = self.node_conv1(x, edge_index)
        x = F.tanh(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.node_conv2(x, edge_index)
        x = F.tanh(x)
        x = self.node_conv3(x, edge_index)
        
        #Edge Feature update
        
        edge_attr = self.edge_conv1(edge_attr, edge_index)
        edge_attr = F.tanh(edge_attr)
        edge_attr = F.dropout(edge_attr, p=0.5, training=self.training)
        edge_attr = self.edge_conv2(edge_attr, edge_index)
        edge_attr = F.tanh(edge_attr)
        edge_attr = self.edge_conv3(edge_attr, edge_index)
        
        
        # Aggregate edge features to nodes
        aggregated_edge_attr = scatter_mean(edge_attr, edge_index[0], dim=0, dim_size=x.shape[0])
        x_transformed = self.node_transform(x)
        
        # # Apply node_transform to node features
        message_input = torch.cat([x_transformed, aggregated_edge_attr], dim=1)
        message_input_transformed = self.message_transform(message_input)
        
        x = x + message_input_transformed
        
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

        # Regressor
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x