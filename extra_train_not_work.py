#!/usr/bin/env python
# extra-train.py — дообучение бинарной модели "<POS> vs not-<POS>"

from __future__ import annotations
import argparse, importlib, inspect, os, warnings, csv
from pathlib import Path
from datetime import datetime
from typing import Sequence, Callable, Any
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import UnidentifiedImageError
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- CLI ----------------
cli = argparse.ArgumentParser("extra-train (continue training on existing weights)")
cli.add_argument("--model",    required=True)                      # alexnet / resnet ...
cli.add_argument("--weights",  required=True)                      # .pt (full or weights-only)
cli.add_argument("--roots",    required=True, help="comma-separated dataset roots (train/ val/ inside)")
cli.add_argument("--pos",      required=True, help="positive class folder name (cat or dog)")
cli.add_argument("--epochs",   type=int, default=30, help="additional epochs to train")
cli.add_argument("--batch-size", type=int, default=32)
cli.add_argument("--lr",       type=float, default=3e-4, help="LR used after resume")
cli.add_argument("--img-size", type=int, default=224)
cli.add_argument("--device",   default="auto", choices=["auto","cpu","cuda","dml"])
cli.add_argument("--save-each", type=int, default=1, help="save full ckpt every N epochs")
cli.add_argument("--outdir",   default=".", help="dir to save checkpoints/logs/plots")
args = cli.parse_args()

# ---------------- device ----------------
if args.device == "cpu":
    device = torch.device("cpu")
elif args.device == "cuda":
    device = torch.device("cuda")
elif args.device == "dml":
    import torch_directml
    device = torch_directml.device()
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml
            device = torch_directml.device()
        except Exception:
            device = torch.device("cpu")

# ---------------- paths & const ----------------
roots: Sequence[Path] = [Path(p.strip()) for p in args.roots.split(",")]
POS_NAME = args.pos.lower()
OUTDIR   = Path(args.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def make_tgt(cls2idx: dict[str,int]) -> Callable[[int],int]:
    pos_ids = {i for n,i in cls2idx.items() if n.lower() == POS_NAME}
    return lambda idx: 1 if idx in pos_ids else 0

class SafeFolder(ImageFolder):
    def __getitem__(self, i):  # type: ignore[override]
        path, tgt = self.samples[i]
        try:
            img = self.loader(path)
        except (UnidentifiedImageError, OSError) as e:
            warnings.warn(f"[skip] {path} — {e}")
            return self.__getitem__((i + 1) % len(self.samples))
        if self.transform:        img = self.transform(img)
        if self.target_transform: tgt = self.target_transform(tgt)
        return img, tgt

def build_concat(split: str, tf):
    parts = []
    for r in roots:
        base = ImageFolder(r / split)
        parts.append(SafeFolder(r / split, transform=tf,
                                target_transform=make_tgt(base.class_to_idx)))
    return ConcatDataset(parts)

# ---------------- datasets ----------------
train_tf = T.Compose([
    T.Resize((args.img_size, args.img_size)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])
val_tf = T.Compose([
    T.Resize((args.img_size, args.img_size)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

train_ds = build_concat("train", train_tf)
val_ds   = build_concat("val",   val_tf)

labels = [y for _, y in train_ds]
cnt    = Counter(labels)
print(f"[info] train dist 0/1 → {cnt}")
pos, neg = cnt[1], cnt[0]
if pos == 0:
    raise SystemExit(f"[fatal] no positive samples '{POS_NAME}' in train set")

# sampler & loaders
sample_w = [1/pos if l == 1 else 1/neg for l in labels]
sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

pos_weight = torch.tensor([neg/pos], device=device)

# ---------------- model ----------------
mod = importlib.import_module(args.model)
Model = (getattr(mod, "CNNModel", None) or
         getattr(mod, "Model", None) or
         next(cls for _, cls in inspect.getmembers(mod, inspect.isclass)
              if issubclass(cls, nn.Module) and cls.__module__ == mod.__name__))
net = Model().to(device)

print(f"[✓] {args.model}.{Model.__name__} POS='{POS_NAME}' "
      f"params={sum(p.numel() for p in net.parameters()):,}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, nesterov=True, weight_decay=1e-6)

# ---------------- load weights ----------------
start_epoch = 1
if not os.path.exists(args.weights):
    raise SystemExit(f"[fatal] weights file not found: {args.weights}")

ckpt: Any = torch.load(args.weights, map_location="cpu")
print("[debug] loaded keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else "state_dict only")

# ---- 1. грузим только веса модели
if isinstance(ckpt, dict) and "model" in ckpt:
    net.load_state_dict(ckpt["model"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    print(f"[info] resume from epoch {start_epoch-1}")
else:
    net.load_state_dict(ckpt, strict=False)
    print("[warn] weights-only file — start_epoch=1")

# ---- 2. ВАЖНО: создаём НОВЫЙ оптимизатор (не восстанавливаем из ckpt)
optimizer = torch.optim.SGD(net.parameters(),
                            lr=args.lr, momentum=0.9,
                            nesterov=True, weight_decay=1e-6)
print(f"[info] fresh optimizer, lr={args.lr:.1e}")


net.to(device)

# ---------------- train loop ----------------
hist = {"tr": [], "vl": [], "ac": []}
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = OUTDIR / f"extra_log_{args.model}_{POS_NAME}_{stamp}.csv"

with open(csv_path, "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["epoch","train_loss","val_loss","val_acc"])
    total_epochs = start_epoch + args.epochs - 1

    for ep in range(start_epoch, total_epochs + 1):
        # ----- train -----
        net.train(); run = 0.0
        for batch_idx, (x, y) in enumerate(tqdm(train_dl, total=len(train_dl),
                                               desc=f"ep {ep}/{total_epochs}", ncols=80)):
            x = x.to(device); y = y.to(device).float()

            optimizer.zero_grad()
            logits = net(x).view(-1)
            loss = criterion(logits, y)
            loss.backward()

            # Диагностика градиента только на самом первом батче после resume
            if ep == start_epoch and batch_idx == 0:
                with torch.no_grad():
                    total_norm = 0.0
                    nnz = 0
                    for p in net.parameters():
                        if p.grad is not None:
                            g = p.grad
                            total_norm += g.norm(2).item() ** 2
                            nnz += (g != 0).sum().item()
                    total_norm = total_norm ** 0.5
                print(f"[dbg] grad_norm_first_batch={total_norm:.4f}, nnz={nnz}, "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

            optimizer.step()
            run += loss.item() * x.size(0)

        tr_loss = run / len(train_dl.dataset)

        # ----- val -----
        net.eval(); vloss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device); y = y.to(device).float()
                lg = net(x).view(-1)
                vloss   += criterion(lg, y).item() * x.size(0)
                correct += ((torch.sigmoid(lg) > 0.5).long() == y.long()).sum().item()
        val_loss = vloss / len(val_dl.dataset)
        val_acc  = correct / len(val_dl.dataset)

        hist["tr"].append(tr_loss); hist["vl"].append(val_loss); hist["ac"].append(val_acc)
        wr.writerow([ep, tr_loss, val_loss, val_acc]); fh.flush()
        print(f"ep {ep}: tr {tr_loss:.3f} | vl {val_loss:.3f} | acc {val_acc:.3f}")

        # save full checkpoint periodically
        if (ep % args.save_each) == 0 or ep == total_epochs:
            full_ckpt = OUTDIR / f"{args.model}_{POS_NAME}_{stamp}_e{ep:03d}.pt"
            torch.save({"model": net.state_dict(),
                        "optim": optimizer.state_dict(),
                        "epoch": ep},
                       full_ckpt)
            print(f"[ckpt] saved -> {full_ckpt}")

# final weights-only
final_w = OUTDIR / f"{args.model}_{POS_NAME}_{stamp}.pt"
torch.save(net.state_dict(), final_w)
print(f"[saved] {final_w}")

# quick plot
plt.plot(hist["tr"], label="train")
plt.plot(hist["vl"], label="val")
plt.plot(hist["ac"], label="acc")
plt.legend(); plt.tight_layout(); plt.show()
