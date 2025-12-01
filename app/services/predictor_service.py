import json
from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn

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

class PredictorService:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[2]
        self.models_dir = self.root / "docs/saved_models"
        self.dataset_dir = self.root / "docs/media/dataset"
        self.hamsters_dir = self.root / "docs/media/img/hamsters"
        self.canonical = ["neutral", "angry", "sad", "happy", "impressed", "sigma"]
        self.alias_to_canonical = {"cry": "sad", "nutral": "neutral"}
        self.canon_to_names = {
            "neutral": ["neutral", "nutral"],
            "angry": ["angry"],
            "sad": ["sad", "cry"],
            "happy": ["happy"],
            "impressed": ["impressed"],
            "sigma": ["sigma"],
        }
        self.images = {l: self._load_img_for_canonical(l) for l in self.canonical}
        self.mean, self.std = self._load_or_compute_norm()
        self.model, self.label_map = self._load_model(in_dim=len(self.mean))
        self.centroids = self._build_centroids()

    def predict_id(self, coords: np.ndarray) -> int:
        x = coords.reshape(-1).astype(np.float32)
        s = np.where(self.std == 0, 1.0, self.std)
        x = (x - self.mean) / s
        if self.model is not None and self.label_map is not None:
            with torch.no_grad():
                t = torch.from_numpy(x).unsqueeze(0)
                logits = self.model(t)
                idx_train = int(torch.argmax(logits, dim=1).item())
            train_label = self.label_map.get(str(idx_train), self.canonical[0])
            canon_label = self.alias_to_canonical.get(train_label, train_label)
            return self.canonical.index(canon_label)
        best_id, best_dist = 0, None
        for i, l in enumerate(self.canonical):
            c = self.centroids.get(l)
            if c is None:
                continue
            d = np.linalg.norm(x - c)
            if best_dist is None or d < best_dist:
                best_dist, best_id = d, i
        return best_id

    def label_for_id(self, idx: int) -> str:
        return self.canonical[idx]

    def image_for_id(self, idx: int):
        return self.images.get(self.canonical[idx])

    def _load_img_for_canonical(self, label):
        for name in self.canon_to_names[label]:
            p = self.hamsters_dir / f"{name}.png"
            img = cv2.imread(str(p))
            if img is not None:
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                return img
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def _load_or_compute_norm(self):
        f = self.models_dir / "norm.npz"
        if f.exists():
            d = np.load(str(f))
            mean = d["mean"].astype(np.float32)
            std = d["std"].astype(np.float32)
            std = np.where(std == 0, 1.0, std)
            return mean, std
        xs = []
        for fp in sorted(self.dataset_dir.glob("*.npy")):
            a = np.load(str(fp)).astype(np.float32).reshape(-1)
            xs.append(a)
        if len(xs) == 0:
            m = np.zeros(1404, dtype=np.float32)
            s = np.ones(1404, dtype=np.float32)
            return m, s
        X = np.stack(xs)
        m = X.mean(axis=0).astype(np.float32)
        s = X.std(axis=0).astype(np.float32)
        s = np.where(s == 0, 1.0, s)
        return m, s

    def _load_model(self, in_dim):
        w = self.models_dir / "landmark_mlp.pt"
        lm = self.models_dir / "labels.json"
        if w.exists() and lm.exists():
            with open(lm, "r") as fh:
                label_map = json.load(fh)
            model = LandmarkMLP(in_dim=in_dim, out_dim=len(label_map))
            sd = torch.load(str(w), map_location="cpu")
            model.load_state_dict(sd)
            model.eval()
            return model, label_map
        return None, None

    def _build_centroids(self):
        groups = {l: [] for l in self.canonical}
        for fp in sorted(self.dataset_dir.glob("*.npy")):
            name = fp.name.split("_")[0]
            canon = self.alias_to_canonical.get(name, name)
            if canon not in groups:
                continue
            a = np.load(str(fp)).astype(np.float32).reshape(-1)
            s = np.where(self.std == 0, 1.0, self.std)
            a = (a - self.mean) / s
            groups[canon].append(a)
        cents = {}
        for l, xs in groups.items():
            if len(xs) == 0:
                continue
            X = np.stack(xs)
            cents[l] = X.mean(axis=0)
        return cents