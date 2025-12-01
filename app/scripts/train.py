import os
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class LandmarkMLP(nn.Module):
    def __init__(self, in_dim=1404, h1=512, h2=256, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_dim),
        )
    def forward(self, x):
        return self.net(x)
    
ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "docs/media/dataset"
HAMSTERS = ROOT / "docs/media/img/hamsters"
MODELS = ROOT / "docs/saved_models"

def load_labels():
    return [p.stem for p in sorted(HAMSTERS.glob("*.png"))]

def load_data(labels):
    X, y = [], []
    for i, l in enumerate(labels):
        for f in DATASET.glob(f"{l}_*.npy"):
            a = np.load(str(f)).astype(np.float32).reshape(-1)
            X.append(a)
            y.append(i)
    return np.stack(X), np.array(y, dtype=np.int64)

def main():
    labels = load_labels()
    X, y = load_data(labels)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    Xn = (X - mean) / std

    os.makedirs(MODELS, exist_ok=True)
    np.savez(str(MODELS / "norm.npz"), mean=mean, std=std)
    with open(MODELS / "labels.json", "w") as fh:
        json.dump({str(i): l for i, l in enumerate(labels)}, fh)

    x = torch.from_numpy(Xn)
    t = torch.from_numpy(y)
    ds = TensorDataset(x, t)
    dl = DataLoader(ds, batch_size=max(4, len(ds)//2), shuffle=True)

    model = LandmarkMLP(in_dim=Xn.shape[1], out_dim=len(labels))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(200):
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1)
        acc = (pred == t).float().mean().item()
        print(f"Train acc: {acc:.3f}")

    torch.save(model.state_dict(), str(MODELS / "landmark_mlp.pt"))

if __name__ == "__main__":
    main()

