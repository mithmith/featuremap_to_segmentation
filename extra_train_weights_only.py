#!/usr/bin/env python
# fine_tune.py — дообучение из weights-only (или полного .pt) без обязательного чекпоинта

from __future__ import annotations
import argparse, importlib, inspect, warnings, os
from pathlib import Path
from datetime import datetime
from typing import Any, Tuple

import torch, torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype, to_tensor
from PIL import Image
from tqdm import tqdm

# ─────────── CLI ───────────
cli = argparse.ArgumentParser("fine-tune from weights (.pt) without full checkpoint")
cli.add_argument("--model",        required=True, help="python module with CNNModel/Model class (resnet, alexnet, ...)")
cli.add_argument("--pos",          required=True, help="cat | dog")
cli.add_argument("--dataset-root", required=True, help="ROOT/{pos,not_pos}/{train,val}/*")
cli.add_argument("--weights",      required=True, help=".pt file with state_dict or full checkpoint")
cli.add_argument("--epochs",       type=int, default=20, help="additional epochs to train")
cli.add_argument("--batch-size",   type=int, default=32)
cli.add_argument("--lr",           type=float, default=3e-4, help="fine-tune learning rate")
cli.add_argument("--img-size",     type=int, default=224)
cli.add_argument("--device",       default="auto", choices=["auto","cpu","cuda","dml"])
cli.add_argument("--outdir",       default="models_ft")
cli.add_argument("--save-each",    type=int, default=1, help="save full checkpoint every N epochs")
args = cli.parse_args()

POS_NAME = args.pos.lower()
NEG_NAME = f"not_{POS_NAME}"
ROOT     = Path(args.dataset_root)
OUTDIR   = Path(args.outdir); OUTDIR.mkdir(parents=True, exist_ok=True)

# ─────────── device ───────────
if args.device == "cpu":
    device = torch.device("cpu")
elif args.device == "cuda":
    device = torch.device("cuda")
elif args.device == "dml":
    import torch_directml
    device = torch_directml.device()
else:  # auto
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml
            device = torch_directml.device()
        except Exception:
            device = torch.device("cpu")

# ─────────── transforms ───────────
train_tf = T.Compose([
    T.Resize((args.img_size, args.img_size)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    # вход уже tensor (см. FlatFolder), тут только нормализация
    T.Normalize([0.5]*3, [0.5]*3),
])
val_tf = T.Compose([
    T.Resize((args.img_size, args.img_size)),
    T.Normalize([0.5]*3, [0.5]*3),
])

# ─────────── FlatFolder с fallback на PIL ───────────
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

class FlatFolder(Dataset):
    """Рекурсивно собирает картинки под root и отдаёт фиксированный label (0/1).
       Быстрый loader: torchvision.io.read_image; fallback: PIL (для «нестандартных» JPEG)."""
    def __init__(self, root: str | Path, transform, fixed_label: int):
        self.root   = Path(root)
        self.paths  = [p for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXT]
        if not self.paths:
            raise FileNotFoundError(f"[fatal] no images inside {self.root}")
        self.transform   = transform
        self.fixed_label = fixed_label

    def __len__(self): return len(self.paths)

    def _load_tensor_rgb(self, path: Path) -> torch.Tensor:
        # 1) быстрый C++-loader
        try:
            img = read_image(str(path))                    # CxHxW (uint8)
            img = convert_image_dtype(img, torch.float32)  # 0..1
        except Exception:
            # 2) Pillow fallback
            pil = Image.open(path).convert("RGB")          # гарантируем 3 канала
            img = to_tensor(pil)                           # 0..1, CxHxW (3,H,W)

        # нормализация каналов к 3 (на всякий)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:
            img = img[:3]
        return img

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = self._load_tensor_rgb(path)
        except Exception as e:
            warnings.warn(f"[skip] {path} — {e}")
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            img = self.transform(img)
        return img, self.fixed_label

# ─────────── datasets & loaders ───────────
def build_split(split: str, tf) -> Tuple[ConcatDataset, int, int]:
    pos_dir = ROOT / POS_NAME / split
    neg_dir = ROOT / NEG_NAME / split
    if not pos_dir.is_dir() or not neg_dir.is_dir():
        raise SystemExit(f"[fatal] expected:\n  {pos_dir}\n  {neg_dir}")

    pos_ds = FlatFolder(pos_dir, tf, fixed_label=1)
    neg_ds = FlatFolder(neg_dir, tf, fixed_label=0)
    ds = ConcatDataset([pos_ds, neg_ds])

    return ds, len(pos_ds), len(neg_ds)

train_ds, n_pos_tr, n_neg_tr = build_split("train", train_tf)
val_ds,   n_pos_vl, n_neg_vl = build_split("val",   val_tf)

# веса для-семплера (без прохода по данным)
labels_train = [1]*n_pos_tr + [0]*n_neg_tr
freq = torch.tensor([n_neg_tr, n_pos_tr], dtype=torch.float32)
weights = 1.0 / freq
sample_weights = [weights[l].item() for l in labels_train]
sampler = WeightedRandomSampler(sample_weights, len(labels_train), replacement=True)

train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

print(f"[info] train counts: pos={n_pos_tr:,}  neg={n_neg_tr:,}")

# ─────────── модель ───────────
mod = importlib.import_module(args.model)
Model = (getattr(mod, "CNNModel", None)
         or getattr(mod, "Model", None)
         or next(cls for _, cls in inspect.getmembers(mod, inspect.isclass)
                 if issubclass(cls, nn.Module) and cls.__module__ == mod.__name__))
net = Model().to(device)

# ─────────── загрузка весов (weights-only ИЛИ full-ckpt) ───────────
weights_path = Path(args.weights)
if not weights_path.exists():
    raise SystemExit(f"[fatal] weights file not found: {weights_path}")
ckpt: Any = torch.load(weights_path, map_location="cpu")

start_epoch = 1
if isinstance(ckpt, dict) and "model" in ckpt:
    net.load_state_dict(ckpt["model"])
    print("[info] loaded full checkpoint dict['model']")
else:
    net.load_state_dict(ckpt, strict=False)
    print("[info] loaded weights-only state_dict")

# ─────────── loss / optim ───────────
pos_weight = torch.tensor([n_neg_tr / max(1, n_pos_tr)], device=device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer  = torch.optim.SGD(
    net.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-6
)

print(f"[✓] {args.model}.{net.__class__.__name__}  params={sum(p.numel() for p in net.parameters()):,}")
print(f"[info] fine-tune lr={args.lr:g}, epochs={args.epochs}")

# ─────────── train loop ───────────
hist = {"tr": [], "vl": [], "ac": []}
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base  = f"{args.model}_{POS_NAME}_{stamp}"

for ep in range(start_epoch, start_epoch + args.epochs):
    # --- train ---
    net.train(); run = 0.0
    for x, y in tqdm(train_dl, total=len(train_dl), ncols=90,
                     desc=f"ep {ep}/{start_epoch+args.epochs-1}"):
        x, y = x.to(device), y.to(device).float()
        optimizer.zero_grad()
        loss = criterion(net(x).view(-1), y)
        loss.backward(); optimizer.step()
        run += loss.item() * x.size(0)
    tr = run / len(train_dl.dataset)

    # --- val ---
    net.eval(); vl, cor = 0.0, 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device).float()
            lg = net(x).view(-1)
            vl += criterion(lg, y).item() * x.size(0)
            cor += ((torch.sigmoid(lg) > 0.5).long() == y.long()).sum().item()
    vl /= len(val_dl.dataset); acc = cor / len(val_dl.dataset)
    hist["tr"].append(tr); hist["vl"].append(vl); hist["ac"].append(acc)
    print(f"ep {ep}: tr {tr:.3f} | vl {vl:.3f} | acc {acc:.3f}")

    # периодически сохраняем полный чекпоинт (на всякий)
    if (ep % args.save_each) == 0 or ep == (start_epoch + args.epochs - 1):
        full_ckpt = OUTDIR / f"{base}_e{ep:03d}.pt"
        torch.save({"model": net.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": ep}, full_ckpt)
        print(f"[ckpt] saved: {full_ckpt}")

# финальные веса-only (state_dict)
final_w = OUTDIR / f"{base}.weights.pt"
torch.save(net.state_dict(), final_w)
print(f"[weights] saved: {final_w}")
