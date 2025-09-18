#!/usr/bin/env python
# feature_int_and_table.py — для каждой картинки:
# (1) PNG-плитка 8×8 из 64 карт признаков выбранного слоя (с нумерацией 0..63)
# (2) интегральная теплокарта (float 0..1) + оверлей поверх исходного

from __future__ import annotations
import argparse, importlib, inspect, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

# ---------------- CLI ----------------
cli = argparse.ArgumentParser("feature-maps grid (64ch) + integral heatmap")
cli.add_argument("--model",    required=True, help="python module with CNNModel/Model class (alexnet, resnet, ...)")
cli.add_argument("--weights",  required=True, help=".pt file: full checkpoint or weights-only")
cli.add_argument("--images-dir", required=True, help="folder with POS images (e.g. .../table_features/cat)")
cli.add_argument("--probe",    required=True,
                 help="layer to hook: 'before_pool3' (auto) or explicit path like 'features[8]'/'layers[8]'")
cli.add_argument("--img-size", type=int, default=224)
cli.add_argument("--device",   default="cpu", choices=["auto","cpu","cuda","dml"])
cli.add_argument("--topk",     type=int, default=64, help="сколько каналов выводить в плитке (обычно 64)")
cli.add_argument("--samples",  type=int, default=16, help="сколько изображений взять из папки")
cli.add_argument("--tiles-out", default=r"D:\progs\work\features_project\fmaps_tiles",
                 help="куда сохранять PNG-плитки (64 карты признаков на картинку)")
cli.add_argument("--int-out",   default=r"D:\progs\work\features_project\int_feature",
                 help="куда сохранять интегральные теплокарты")
# ПРОЗРАЧНОСТЬ ДЛЯ НАЛОЖЕНИЯ КАРТ ПРИЗНАКОВ
cli.add_argument("--feat-alpha", type=float, default=0.6,
                 help="прозрачность карты признаков на плитке (0..1)")
args = cli.parse_args()

# ---------------- device ----------------
def pick_device():
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        return torch.device("cuda")
    if args.device == "dml":
        import torch_directml
        return torch_directml.device()
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")

device = pick_device()

# ---------------- utils ----------------
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

def find_convs_before_pools(model: nn.Module) -> Dict[str, Tuple[str, nn.Module]]:
    out: Dict[str, Tuple[str, nn.Module]] = {}
    last_conv_name = None
    last_conv_mod: nn.Module | None = None
    k = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv_name, last_conv_mod = name, m
        if isinstance(m, nn.MaxPool2d) and last_conv_mod is not None:
            k += 1
            out[f"before_pool{k}"] = (last_conv_name, last_conv_mod)
            last_conv_name, last_conv_mod = None, None
    if last_conv_mod is not None and last_conv_name is not None:
        out["last_conv"] = (last_conv_name, last_conv_mod)
    return out

def normalize_01(t: torch.Tensor) -> torch.Tensor:
    # простая обрезка диапазона до [0,1] — без min-max
    return t.clamp(0, 1)

def denorm_img(x: torch.Tensor) -> np.ndarray:
    # x: 1×3×H×W (нормировано под mean=0.5 std=0.5)
    im = (x.squeeze(0) * 0.5 + 0.5).clamp(0,1).permute(1,2,0).cpu().numpy()
    return im

# ---------------- load model ----------------
mod = importlib.import_module(args.model)
Model = resolve_model_class(mod)
net = Model().to(device).eval()

ckpt = torch.load(args.weights, map_location="cpu")
if isinstance(ckpt, dict) and "model" in ckpt:
    net.load_state_dict(ckpt["model"])
else:
    net.load_state_dict(ckpt, strict=False)

MODEL_LABEL = f"{args.model}.{net.__class__.__name__}"

# определить слой для хуков
if args.probe.startswith("before_pool"):
    mapping = find_convs_before_pools(net)
    if args.probe not in mapping:
        avail = ", ".join(mapping.keys()) if mapping else "<none>"
        raise SystemExit(f"[fatal] probe '{args.probe}' not found. Available: {avail}")
    probe_path, probe_layer = mapping[args.probe]
else:
    probe_path = args.probe
    probe_layer = get_module_by_path(net, args.probe)

print(f"[info] probe layer: {args.probe} -> {probe_path}")

# ---------------- images ----------------
IMG_DIR = Path(args.images_dir)
if not IMG_DIR.is_dir():
    raise SystemExit(f"[fatal] images-dir not found: {IMG_DIR}")

exts = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
all_paths = [p for p in sorted(IMG_DIR.iterdir()) if p.suffix.lower() in exts and p.is_file()]
if not all_paths:
    raise SystemExit(f"[fatal] no images in {IMG_DIR}")

sel_paths = all_paths[:args.samples]
print(f"[info] images selected: {len(sel_paths)}")

# ---------------- transforms ----------------
tf = T.Compose([
    T.Resize((args.img_size, args.img_size)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def load_tensor(path: Path) -> torch.Tensor | None:
    try:
        img = Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        warnings.warn(f"[skip] {path} — {e}")
        return None
    return tf(img).unsqueeze(0)

# ---------------- forward hook ----------------
fmaps: Dict[str, torch.Tensor] = {}
def hook(_m, _i, out):
    fmaps["probe"] = out.detach()

handle = probe_layer.register_forward_hook(hook)

# ---------------- out dirs ----------------
TILES_DIR = Path(args.tiles_out); TILES_DIR.mkdir(parents=True, exist_ok=True)
INT_DIR   = Path(args.int_out);   INT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- main loop ----------------
TOPK = args.topk

for p in sel_paths:
    xt = load_tensor(p)
    if xt is None:
        continue

    xt = xt.to(device)
    with torch.no_grad():
        _ = net(xt)

    if "probe" not in fmaps:
        warnings.warn(f"[warn] no fmap captured for {p.name}")
        continue

    fmap = fmaps["probe"].cpu().squeeze(0)   # C × h × w
    C, h, w = fmap.shape
    K = min(C, TOPK)

    # апскейл до img_size и «обрезка» 0..1
    up = F.interpolate(fmap.unsqueeze(0), size=(args.img_size, args.img_size),
                       mode="bilinear", align_corners=False).squeeze(0)  # C × H × W
    ch_maps = [normalize_01(up[ch]) for ch in range(K)]  # список H×W (0..1)

    # подготовим базовое изображение (денормализованное)
    base_np = denorm_img(xt.cpu())  # H×W×3, 0..1

    # --- PNG ПЛИТКА 8×8 С НАЛОЖЕНИЕМ ФОТО ---
    grid_n = int(np.ceil(np.sqrt(K)))
    fig, axes = plt.subplots(grid_n, grid_n, figsize=(grid_n*2.4, grid_n*2.4))
    axes = np.asarray(axes)
    for i in range(grid_n*grid_n):
        r, c = divmod(i, grid_n)
        ax = axes[r, c]
        ax.axis("off")
        if i < K:
            # Сначала фото, затем полупрозрачная карта признака
            ax.imshow(base_np, alpha=1.0)
            ax.imshow(ch_maps[i].numpy(), cmap="magma", vmin=0, vmax=1,
                      alpha=args.feat_alpha)
            ax.text(3, 10, f"{i}", color="w", fontsize=9,
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.5))
    fig.suptitle(f"{p.stem}   |   {MODEL_LABEL}   |   {probe_path}",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    tile_name = f"{p.stem}__{args.model}_64ch.png"
    fig.savefig(TILES_DIR / tile_name, dpi=180)
    plt.close(fig)

    # --- ИНТЕГРАЛЬНАЯ КАРТА (float 0..1, без бинаризации) ---
    integral = torch.stack(ch_maps, dim=0).sum(0)  # H×W
    integral = normalize_01(integral)

    raw_path = INT_DIR / f"{p.stem}__{args.model}__integral.png"
    overlay_path = INT_DIR / f"{p.stem}__{args.model}__integral_overlay.png"
    plt.imsave(raw_path, integral.numpy(), cmap="magma")

    plt.figure(figsize=(4.5,4.5))
    plt.imshow(base_np)
    plt.imshow(integral.numpy(), cmap="jet", alpha=0.45)
    plt.title(f"{p.stem} | integral | {MODEL_LABEL}")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(overlay_path, dpi=180); plt.close()

    print(f"[ok] {p.name}: tiles → {tile_name}, integral → {overlay_path.name}")

# cleanup
handle.remove()
print("[done]")
