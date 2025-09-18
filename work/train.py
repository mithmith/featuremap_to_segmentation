# train.py
import argparse
import csv
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─── Device ───────────────────────────────────────────────────────────────────
def get_device(name: str) -> torch.device:
    name = name.lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "dml":
        try:
            import torch_directml  # noqa: F401
            return torch_directml.device()
        except Exception as e:
            tqdm.write(f"[warn] torch-directml not available ({e}); falling back to CPU.")
    return torch.device("cpu")


# ─── Transforms & Dataloaders ─────────────────────────────────────────────────
def _fallback_transforms(img_size: int):
    from torchvision import transforms as T
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, val_tf


def _safe_imagefolder(root: Path, split: str, transform, pos: str):
    from torchvision.datasets import ImageFolder
    from PIL import Image, UnidentifiedImageError

    class SafeIF(ImageFolder):
        def __init__(self, root, **kw):
            super().__init__(root, **kw)
            self._idx2name = {v: k for k, v in self.class_to_idx.items()}
            self._pos = pos

        def __getitem__(self, i):
            p, t = self.samples[i]
            try:
                x = self.loader(p)
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError):
                return None
            if self.transform is not None:
                x = self.transform(x)
            y = 1 if self._idx2name[t] == self._pos else 0
            return x, y

    return SafeIF(root / split, transform=transform)


def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def build_dataloaders(dataset_root: str, pos: str, img_size: int,
                      batch_size: int, num_workers: int, pin_memory: bool) -> Tuple[DataLoader, DataLoader]:
    # Prefer user's dataset.py if present
    try:
        import dataset as ds  # type: ignore
        if hasattr(ds, "build_dataloaders"):
            return ds.build_dataloaders(dataset_root, pos, img_size, batch_size, num_workers, pin_memory)
    except Exception as e:
        tqdm.write(f"[warn] dataset.build_dataloaders failed, using fallback: {e}")

    # Build internally
    try:
        from transform import build_transforms  # type: ignore
        train_tf, val_tf = build_transforms(img_size)
    except Exception:
        train_tf, val_tf = _fallback_transforms(img_size)

    root = Path(dataset_root)
    train_ds = _safe_imagefolder(root, "train", train_tf, pos)
    val_ds   = _safe_imagefolder(root, "val",   val_tf,  pos)

    train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers,
                          pin_memory=pin_memory, collate_fn=_collate_skip_none)
    val_dl   = DataLoader(val_ds,   batch_size, False, num_workers=num_workers,
                          pin_memory=pin_memory, collate_fn=_collate_skip_none)
    return train_dl, val_dl


# ─── Model / Utils ────────────────────────────────────────────────────────────
def load_model(module: str) -> nn.Module:
    m = __import__(module)
    cls = getattr(m, "Model", None) or getattr(m, "CNNModel", None)
    if cls is None:
        raise RuntimeError(f"{module} must export Model or CNNModel")
    return cls()


def logits_1d(out: torch.Tensor) -> torch.Tensor:
    if out.dim() == 2 and out.size(1) == 1:
        return out[:, 0]
    if out.dim() == 1:
        return out
    raise RuntimeError(f"Unexpected output shape {tuple(out.shape)}; expected (B,1) or (B,)")


@torch.no_grad()
def eval_epoch(model: nn.Module, dl: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    tot_loss = tot = correct = 0
    for batch in tqdm(dl, desc="val", leave=False):
        if batch is None:
            continue
        x, y = batch
        x, y = x.to(device), y.float().to(device)
        lg = logits_1d(model(x))
        loss = crit(lg, y)
        tot_loss += loss.item() * x.size(0)
        correct += (torch.sigmoid(lg).ge(0.5).long() == y.long()).sum().item()
        tot += x.size(0)
    return tot_loss / max(1, tot), correct / max(1, tot)


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    import numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)


def sd_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


# ─── Train ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("Train binary classifier (pos vs background)")
    ap.add_argument("--model", choices=["alexnet", "resnet"], required=True)
    ap.add_argument("--pos", choices=["cat", "dog"], required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--outdir", default=r"D:\progs\work\train_model")
    ap.add_argument("--device", choices=["dml", "cuda", "cpu"], default="dml")
    ap.add_argument("--resume-weights", default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    tqdm.write(f"[info] device: {device}")

    train_dl, val_dl = build_dataloaders(args.dataset_root, args.pos, args.img_size,
                                         args.batch_size, args.num_workers, pin_memory=False)

    model = load_model(args.model).to(device)

    resume_path = args.resume_weights
    if resume_path:
        tqdm.write(f"[info] loading {resume_path}")
        inc = model.load_state_dict(torch.load(resume_path, map_location="cpu"), strict=False)
        try:
            tqdm.write(f"[info] missing={list(inc.missing_keys)} unexpected={list(inc.unexpected_keys)}")
        except Exception:
            pass

    crit = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = f"{args.model}_{args.pos}_{stamp}"
    csv_path = outdir / f"training_log_{run}.csv"
    weights_path = outdir / f"{run}.pt"

    tr_losses, va_losses, va_accs = [], [], []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        f.flush(); os.fsync(f.fileno())

        epbar = tqdm(range(1, args.epochs + 1), desc="epochs", position=0)
        for ep in epbar:
            model.train()
            tot_loss = tot = 0
            bbar = tqdm(train_dl, desc=f"train {ep}/{args.epochs}", position=1, leave=False)
            for batch in bbar:
                if batch is None:
                    continue
                x, y = batch
                x, y = x.to(device), y.float().to(device)
                opt.zero_grad(set_to_none=True)
                lg = logits_1d(model(x))
                loss = crit(lg, y)
                loss.backward(); opt.step()
                tot_loss += loss.item() * x.size(0); tot += x.size(0)
                bbar.set_postfix(running_loss=f"{tot_loss/max(1,tot):.4f}")
            bbar.close()

            tr = tot_loss / max(1, tot)
            vl, va = eval_epoch(model, val_dl, device)
            tr_losses.append(tr); va_losses.append(vl); va_accs.append(va)

            wr.writerow([ep, f"{tr:.6f}", f"{vl:.6f}", f"{va:.6f}"])
            f.flush(); os.fsync(f.fileno())
            epbar.set_postfix(train_loss=f"{tr:.4f}", val_loss=f"{vl:.4f}", val_acc=f"{va:.4f}")
        epbar.close()

    torch.save(sd_cpu(model.state_dict()), weights_path)
    tqdm.write(f"[✓] saved final weights → {weights_path}")

    # Curves
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 5), dpi=160)
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(tr_losses)+1), tr_losses, label="train loss")
        ax.plot(range(1, len(va_losses)+1), va_losses, label="val loss")
        ax2 = ax.twinx(); ax2.plot(range(1, len(va_accs)+1), va_accs, linestyle="--", label="val acc")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax2.set_ylabel("acc")
        ax.grid(True, alpha=0.3); fig.legend(loc="lower left")
        fig.suptitle(f"{args.model.upper()} | pos={args.pos} | {stamp}")
        png_path = outdir / f"curves_{run}.png"
        fig.tight_layout(); fig.savefig(png_path); plt.close(fig)
        tqdm.write(f"[✓] curves saved to {png_path}")
    except Exception as e:
        tqdm.write(f"[warn] could not save curves: {e}")

    tqdm.write(f"[✓] csv saved → {csv_path}")
    tqdm.write(f"[✓] artifacts dir → {outdir}")


if __name__ == "__main__":
    main()
