import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool,TransformerConv, MeanAggregation

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = torch.nn.Linear(64, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Global pooling
        x = self.fc(x)
        return x
  
class SingleResolutionGraphModel(torch.nn.Module):
    def __init__(self, num_features, latent_dim, num_classes):
        super(SingleResolutionGraphModel, self).__init__()
        hidden_dim = 128
        heads_num = 2
        # Define TransformerConv layers for each resolution graph
        self.conv_20x = nn.Sequential(
            TransformerConv(in_channels=num_features, out_channels=hidden_dim, heads=heads_num, concat=True),
            TransformerConv(in_channels=hidden_dim*heads_num, out_channels=latent_dim, heads=heads_num, concat=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(latent_dim*heads_num, latent_dim//2), #24-8-2
            # nn.ReLU(),
            nn.Linear(latent_dim//2, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)  # Softmax activation f

    def forward(self, data, return_attn=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # x = self.conv_5x[0](x, edge_index)
        x = self.conv_20x[0](x, edge_index)

        x = F.relu(x)
        # x = self.conv_5x[1](x, edge_index)
        x, attn = self.conv_20x[1](x, edge_index, return_attention_weights=return_attn)

        x = F.relu(x)
        x = global_max_pool(x, batch)  # Global pooling
        x = self.fc(x)
        if return_attn:
            return (x, attn)
        return x
    

class MultiResolutionGraphModel(nn.Module):
    def __init__(self, num_features, latent_dim, num_classes):
        super(MultiResolutionGraphModel, self).__init__()
        hidden_dim = 128
        heads_num = 8
        # Define TransformerConv layers for each resolution graph
        self.conv_5x = nn.Sequential(
            TransformerConv(in_channels=num_features, out_channels=hidden_dim, heads=heads_num, concat=True),
            TransformerConv(in_channels=hidden_dim*heads_num, out_channels=latent_dim, heads=heads_num, concat=True)
        )
        self.conv_10x = nn.Sequential(
            TransformerConv(in_channels=num_features, out_channels=hidden_dim, heads=heads_num, concat=True),
            TransformerConv(in_channels=hidden_dim*heads_num, out_channels=latent_dim, heads=heads_num, concat=True)
        )
        self.conv_20x = nn.Sequential(
            TransformerConv(in_channels=num_features, out_channels=hidden_dim, heads=heads_num, concat=True),
            TransformerConv(in_channels=hidden_dim*heads_num, out_channels=latent_dim, heads=heads_num, concat=True)
        )
        
        self.mlp_5x = nn.Linear(in_features=latent_dim*heads_num, out_features=8)
        self.mlp_10x = nn.Linear(in_features=latent_dim*heads_num, out_features=8)
        self.mlp_20x = nn.Linear(in_features=latent_dim*heads_num, out_features=8)

        # MLP for final prediction
        self.fc = nn.Sequential(
            nn.Linear(latent_dim*3, latent_dim), #24-8-2
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(latent_dim, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for classification
    
    def forward(self, data_5x, data_10x, data_20x):
        x_5x = self.conv_5x[0](data_5x.x, data_5x.edge_index)
        x_5x = F.relu(x_5x)
        x_5x = self.conv_5x[1](x_5x, data_5x.edge_index)
        x_5x = F.relu(x_5x)

        x_10x = self.conv_10x[0](data_10x.x, data_10x.edge_index)
        x_10x = F.relu(x_10x)
        x_10x = self.conv_10x[1](x_10x, data_10x.edge_index)
        x_10x = F.relu(x_10x)

        x_20x = self.conv_20x[0](data_20x.x, data_20x.edge_index)
        x_20x = F.relu(x_20x)
        x_20x = self.conv_20x[1](x_20x, data_20x.edge_index)
        x_20x = F.relu(x_20x)

         # Global mean pooling to aggregate features from each graph
        x_5x = global_max_pool(x_5x, data_5x.batch)
        x_10x = global_max_pool(x_10x, data_10x.batch)
        x_20x = global_max_pool(x_20x, data_20x.batch)

        x_5x = self.mlp_5x(x_5x)   
        x_10x = self.mlp_10x(x_10x)   
        x_20x = self.mlp_20x(x_20x)   

        # Concatenate all features
        x = torch.cat([x_5x, x_10x, x_20x], dim=1)
        
        x = self.fc(x)
        return x
