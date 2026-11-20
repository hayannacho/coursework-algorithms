import torch
import torch.nn as nn

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {a: i for i, a in enumerate(amino_acids)}


def one_hot_encode(sequence):
    if not sequence or set(sequence) - set(amino_acids):
        raise ValueError("nonempty standard amino-acid sequence required")
    L = len(sequence)
    tensor = torch.zeros(L, 20, dtype=torch.float32)
    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            tensor[i, aa_to_idx[aa]] = 1
    return tensor


class AttentionClassifier(nn.Module):

    def __init__(self, num_classes, embed_dim=96, num_heads=2, num_layers=2, ff_dim=192):
        super(AttentionClassifier, self).__init__()
        self.embedding = nn.Linear(20, embed_dim)
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.ff_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
                for _ in range(num_layers)
            ]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, seq_lens):
        seq_lens = seq_lens.to(x.device)
        if (
            x.ndim != 3
            or x.shape[2] != 20
            or seq_lens.shape != (x.shape[0],)
            or torch.any(seq_lens < 1)
            or torch.any(seq_lens > x.shape[1])
        ):
            raise ValueError("invalid sequence lengths or input shape")
        x = self.embedding(x)
        max_len = x.shape[1]
        mask = torch.arange(max_len, device=x.device).expand(
            len(seq_lens), max_len
        ) >= seq_lens.unsqueeze(1)
        mask = mask.to(x.device)
        for attn, ff, norm in zip(self.attention_layers, self.ff_layers, self.norm_layers):
            attn_output, _ = attn(x, x, x, key_padding_mask=mask)
            x = norm(x + attn_output)
            ff_output = ff(x)
            x = norm(x + ff_output)
        x = x.masked_fill(mask.unsqueeze(-1), 0).sum(dim=1) / seq_lens.unsqueeze(1)
        return self.fc(x)


def demo():
    torch.manual_seed(42)
    sequences = ["ACDEFG", "AAAAAA", "KKKKKK", "KLMNPQ"]
    x = torch.stack([one_hot_encode(s) for s in sequences])
    lengths = torch.tensor([6] * 4)
    labels = torch.tensor([0, 0, 1, 1])
    model = AttentionClassifier(2, embed_dim=16, num_heads=2, num_layers=1, ff_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    initial = float(criterion(model(x, lengths), labels).detach())
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(x, lengths), labels)
        loss.backward()
        optimizer.step()
    final = float(criterion(model(x, lengths), labels).detach())
    return {
        "data": "four synthetic amino-acid strings; arbitrary labels",
        "initial_loss": initial,
        "final_loss": final,
        "output_shape": list(model(x, lengths).shape),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(demo(), indent=2))
