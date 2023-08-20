import torch
import torch.nn as nn
import torch.nn.functional as F

class LogicalAttentionModule(nn.Module):
    def __init__(self, input_dim, num_connectives):
        super(LogicalAttentionModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, num_connectives)

    def forward(self, z_X):
        x = F.relu(self.fc1(z_X))
        logits = self.fc2(x)
        P_conn = F.softmax(logits, dim=-1)
        return P_conn

# Example usage:
input_dim = 768  # The dimension of the cross-modal embeddings (z_X)
num_connectives = 5  # The number of different logical connectives

lATT = LogicalAttentionModule(input_dim, num_connectives)

# Sample cross-modal embeddings, with shape (batch_size, input_dim)
z_X = torch.randn(32, input_dim)

# Forward pass to get the probabilities of logical connectives
P_conn = lATT(z_X)



