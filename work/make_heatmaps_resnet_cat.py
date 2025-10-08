# Шаг 1: получаем суммарные теплокарты с указанного слоя и сохраняем на диск.
import argparse, importlib, inspect, os
from pathlib import Path
from typing import Any, List, Tuple, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- CLI --------------------
ap = argparse.ArgumentParser("Make summed heatmaps from a given layer (e.g., block4.bn2) for resnet_cat.")
ap.add_argument("--weights", required=True, help="Path to .pt state_dict (e.g., resnet_cat_YYYYMMDD_HHMMSS.pt)")
ap.add_argument("--images-dir", default=r"D:\progs\work\datasets\dataset_for_cat\cat\train",
                help="Directory with input images")
ap.add_argument("--out-dir",    default=r"D:\progs\work\heatmaps_resnet_cat",
                help="Where to save heatmaps")
ap.add_argument("--model",      default="resnet", choices=["resnet"], help="Model module to import")
ap.add_argument("--layer",      default="block4.bn2", help="Module path to hook (e.g., block4.bn2)")
ap.add_argument("--img-size",   type=int, default=256)
ap.add_argument("--device",     choices=["cpu","cuda","dml","auto"], default="auto")
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--num-workers", type=int, default=0)
ap.add_argument("--apply-relu", action="store_true", default=True,
                help="Apply ReLU to feature maps before summation (recommended)")
args = ap.parse_args()

# -------------------- device --------------------
def get_device(which: str) -> torch.device:
    if which == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml  # noqa: F401
            return torch_directml.device()
        except Exception:
            return torch.device("cpu")
    if which == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if which == "dml":
        try:
            import torch_directml  # noqa: F401
            return torch_directml.device()
        except Exception:
            print("[warn] torch-directml not available; falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")

device = get_device(args.device)

# -------------------- transforms --------------------
# Предпочитаем ваш transform.py; иначе — внутренний fallback, эквивалентный evaluate.py (val-трансформ).
# (Resize -> CenterCrop -> ToTensor -> Normalize with ImageNet stats)
def build_val_transform(img_size: int):
    try:
        from transform import build_val_transform as _bv  # type: ignore
        return _bv(img_size)
    except Exception:
        from torchvision import transforms as T
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        return T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

tf_val = build_val_transform(args.img_size)

# -------------------- data --------------------
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

class DirDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, transform):
        self.paths = [p for p in sorted(root.rglob("*"))
                      if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return None  # will be filtered by collate_fn
        x = self.transform(img)
        return x, str(path)

def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# -------------------- model --------------------
def resolve_model_class(mod) -> type[nn.Module]:
    return (getattr(mod, "CNNModel", None)
            or getattr(mod, "Model", None)
            or next(cls for _, cls in inspect.getmembers(mod, inspect.isclass)
                    if issubclass(cls, nn.Module) and cls.__module__ == mod.__name__))

def get_module_by_path(root: nn.Module, path: str) -> nn.Module:
    obj: Any = root
    for token in path.split("."):
        if "[" in token and token.endswith("]"):
            base, idx = token[:-1].split("[")
            obj = getattr(obj, base)
            obj = obj[int(idx)]
        else:
            obj = getattr(obj, token)
    if not isinstance(obj, nn.Module):
        raise ValueError(f"'{path}' is not a nn.Module")
    return obj

mod = importlib.import_module(args.model)  # 'resnet'
Model = resolve_model_class(mod)
net = Model().to(device).eval()

state = torch.load(args.weights, map_location="cpu")
if isinstance(state, dict) and "model" in state:
    missing = net.load_state_dict(state["model"], strict=False)
else:
    missing = net.load_state_dict(state, strict=False)
# print useful info if present
try:
    print(f"[info] missing={list(missing.missing_keys)} unexpected={list(missing.unexpected_keys)}")
except Exception:
    pass

probe_layer = get_module_by_path(net, args.layer)  # e.g., block4.bn2  (exists in your ResNetLike)  # :contentReference[oaicite:4]{index=4}
print(f"[info] hook layer: {args.layer} -> {probe_layer.__class__.__name__}")

# -------------------- hook --------------------
_fmap: Optional[torch.Tensor] = None
def _hook(_m, _in, out):
    global _fmap
    _fmap = out.detach()

handle = probe_layer.register_forward_hook(_hook)

# -------------------- I/O --------------------
in_dir  = Path(args.images_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

if not in_dir.is_dir():
    raise SystemExit(f"[fatal] images-dir not found: {in_dir}")

# -------------------- loader --------------------
ds = DirDataset(in_dir, tf_val)
dl = torch.utils.data.DataLoader(
    ds, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=(device.type != "cpu"),
    collate_fn=_collate_skip_none
)

# -------------------- helpers --------------------
def minmax01(t: torch.Tensor) -> torch.Tensor:
    tmin = torch.amin(t)
    tmax = torch.amax(t)
    denom = (tmax - tmin).clamp_min(1e-12)
    return (t - tmin) / denom

def save_heatmaps(batch_paths: List[str], heat: torch.Tensor):
    """
    batch_paths: list of original image paths, len=B
    heat: B×H×W, float32 in [0,1]
    Saves: *.heat.npy, *.heat.gray.png, *.heat.magma.png
    """
    import matplotlib.pyplot as plt
    heat_np = heat.cpu().numpy()
    for i, p in enumerate(batch_paths):
        stem = Path(p).stem
        npy_path   = out_dir / f"{stem}.heat.npy"
        gray_path  = out_dir / f"{stem}.heat.gray.png"
        magma_path = out_dir / f"{stem}.heat.magma.png"

        # .npy
        np.save(npy_path, heat_np[i])

        # grayscale
        Image.fromarray((heat_np[i] * 255.0).astype(np.uint8), mode="L").save(gray_path)

        # colored
        plt.figure(figsize=(3,3), dpi=160)
        plt.imshow(heat_np[i], cmap="magma", vmin=0.0, vmax=1.0)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(magma_path, bbox_inches="tight", pad_inches=0)
        plt.close()

# -------------------- main loop --------------------
from tqdm import tqdm
with torch.no_grad():
    pbar = tqdm(dl, desc="heatmaps", leave=False)
    for batch in pbar:
        if batch is None:  # all samples in this batch were bad
            continue
        x, paths = batch
        x = x.to(device, non_blocking=True)

        # forward (hook fills _fmap)
        _ = net(x)
        fmap = _fmap
        if fmap is None:
            print("[warn] hook produced no feature maps; skipping batch")
            continue

        # fmap: B×C×h×w  (bn2 output). Optionally ReLU.
        if args.apply_relu:
            fmap = F.relu(fmap)

        # sum over channels -> B×h×w
        summed = fmap.sum(dim=1)

        # upsample to img_size×img_size
        H = W = args.img_size
        up = F.interpolate(summed.unsqueeze(1), size=(H, W), mode="bilinear",
                           align_corners=False).squeeze(1)

        # min-max to [0,1] per-image
        # (vectorized normalization: per-item)
        B = up.shape[0]
        up_flat = up.view(B, -1)
        vmin = up_flat.min(dim=1)[0].view(B, 1, 1)
        vmax = up_flat.max(dim=1)[0].view(B, 1, 1)
        heat01 = (up - vmin) / (vmax - vmin + 1e-12)
        heat01 = heat01.clamp_(0.0, 1.0).float()

        save_heatmaps(list(paths), heat01)

handle.remove()
print(f"[✓] done. Saved heatmaps to: {out_dir}")
