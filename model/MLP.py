import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        super(MLPModel, self).__init__()
        layers = []
        
        # Add the first hidden layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))  # Dropout for regularization

        # Add additional hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Dropout for regularization

        # Add the output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # No activation here; apply sigmoid in the loss function if using BCEWithLogitsLoss

        # Use nn.Sequential to stack the layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)  # Ensure output is of shape [batch_size]
