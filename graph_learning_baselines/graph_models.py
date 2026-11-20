import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class MLPNodeClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPNodeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class GNNNodeClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=2):
        super(GNNNodeClassifier, self).__init__()
        self.num_gcn_layers = num_gcn_layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.fc(x)
        return x


class GCNGraphClassifier(nn.Module):

    def __init__(self, input_dim, mapped_dim, hidden_dim, output_dim, num_gcn_layers=6):
        super(GCNGraphClassifier, self).__init__()
        self.feature_transform = nn.Linear(input_dim, mapped_dim)
        self.conv1 = GCNConv(mapped_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.feature_transform(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


def demo():
    torch.manual_seed(42)
    x = torch.randn(12, 4)
    edges = torch.tensor(
        [
            [i for i in range(12) for _ in range(2)],
            [(i + d) % 12 for i in range(12) for d in (-1, 1)],
        ],
        dtype=torch.long,
    )
    labels = (x[:, 0] > 0).long()
    models = [
        MLPNodeClassifier(4, 8, 2),
        GNNNodeClassifier(4, 8, 2),
        GCNGraphClassifier(4, 8, 8, 2, 2),
    ]
    results = {}
    for model in models:
        args = (
            (x,)
            if isinstance(model, MLPNodeClassifier)
            else (
                (x, edges, torch.zeros(12, dtype=torch.long))
                if isinstance(model, GCNGraphClassifier)
                else (x, edges)
            )
        )
        y = torch.tensor([1]) if isinstance(model, GCNGraphClassifier) else labels
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        initial = float(F.cross_entropy(model(*args), y).detach())
        for _ in range(20):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(*args), y)
            loss.backward()
            optimizer.step()
        final = float(F.cross_entropy(model(*args), y).detach())
        results[type(model).__name__] = {
            "initial_loss": initial,
            "final_loss": final,
            "output_shape": list(model(*args).shape),
        }
    return results


if __name__ == "__main__":
    import json

    print(json.dumps(demo(), indent=2))
