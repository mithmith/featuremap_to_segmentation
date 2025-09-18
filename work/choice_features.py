# choice_features.py
import argparse
import importlib
import inspect
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser("Overlay (Σpos − Σneg) feature maps on images; output a 4×4 tile.")
p.add_argument("--model", required=True, help="Module with Model/CNNModel class (alexnet, resnet, ...)")
p.add_argument("--weights", required=True, help="Path to .pt weights (state_dict)")
p.add_argument("--images-dir", default=r"D:\progs\work\table_features\cat",
               help="Directory with images to visualize (will take first --samples files)")
p.add_argument("--probe", required=True,
               help="Layer to hook: e.g. 'before_pool3' or explicit module path like 'features[8]'/'layer4[1].bn2'")
p.add_argument("--img-size", type=int, default=224)
p.add_argument("--samples", type=int, default=16, help="How many images to include in the 4×4 tile")
p.add_argument("--device", choices=["auto", "cpu", "cuda", "dml"], default="auto")
p.add_argument("--feat-alpha", type=float, default=0.6, help="Overlay transparency (0..1)")
p.add_argument("--out", default=r"D:\progs\work\features_project\fmaps_tiles", help="Output directory for the tile PNG")
p.add_argument("--outfile", default=None, help="Optional filename for the tile PNG")
p.add_argument("--features-pos", nargs="*", default=[], help="Positive feature indices (e.g. 6 11 12 or [6,11,12])")
p.add_argument("--features-neg", nargs="*", default=[], help="Negative feature indices (e.g. 25 29 or [25,29])")
args = p.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────
def pick_device() -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        return torch.device("cuda")
    if args.device == "dml":
        try:
            import torch_directml  # noqa: F401
            return torch_directml.device()
        except Exception as e:
            print(f"[warn] torch-directml not available ({e}); falling back to CPU")
            return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml  # noqa: F401
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


device = pick_device()


# ── Utils ─────────────────────────────────────────────────────────────────────
def resolve_model_class(mod) -> type[nn.Module]:
    return (getattr(mod, "Model", None)
            or getattr(mod, "CNNModel", None)
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
    last_conv = None
    last_name = None
    k = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv, last_name = m, name
        if isinstance(m, nn.MaxPool2d) and last_conv is not None:
            k += 1
            out[f"before_pool{k}"] = (last_name, last_conv)
            last_conv, last_name = None, None
    if last_conv is not None:
        out["last_conv"] = (last_name, last_conv)  # type: ignore[arg-type]
    return out


def parse_idx_list(tokens: List[str]) -> List[int]:
    s = " ".join(tokens).replace("[", " ").replace("]", " ").replace(",", " ")
    out: List[int] = []
    for t in s.split():
        try:
            out.append(int(t))
        except ValueError:
            pass
    return sorted(set(out))


def minmax01(t: torch.Tensor) -> torch.Tensor:
    tmin, tmax = torch.min(t), torch.max(t)
    if (tmax - tmin).abs().item() < 1e-12:
        return torch.zeros_like(t)
    return (t - tmin) / (tmax - tmin)


def denorm_img(x: torch.Tensor) -> np.ndarray:
    # x: 1×3×H×W normalized with mean=0.5 std=0.5
    return (x.squeeze(0).mul(0.5).add(0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy())


# ── Load model & probe ────────────────────────────────────────────────────────
mod = importlib.import_module(args.model)
Model = resolve_model_class(mod)
net = Model().to(device).eval()

state = torch.load(args.weights, map_location="cpu")
if isinstance(state, dict) and "model" in state:
    net.load_state_dict(state["model"], strict=False)
else:
    net.load_state_dict(state, strict=False)

if args.probe.startswith("before_pool"):
    mapping = find_convs_before_pools(net)
    if args.probe not in mapping:
        avail = ", ".join(mapping.keys()) if mapping else "<none>"
        raise SystemExit(f"[fatal] probe '{args.probe}' not found. Available: {avail}")
    probe_path, probe_layer = mapping[args.probe]
else:
    probe_path, probe_layer = args.probe, get_module_by_path(net, args.probe)

print(f"[info] device={device} | probe={args.probe} -> {probe_path}")

# ── Images ────────────────────────────────────────────────────────────────────
img_dir = Path(args.images_dir)
if not img_dir.is_dir():
    raise SystemExit(f"[fatal] images-dir not found: {img_dir}")

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
paths = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts and p.is_file()]
if not paths:
    raise SystemExit(f"[fatal] no images in {img_dir}")
paths = paths[: args.samples]

tf = T.Compose([
    T.Resize((args.img_size, args.img_size)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def load_tensor(path: Path) -> torch.Tensor | None:
    try:
        img = Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        warnings.warn(f"[skip] {path} — {e}")
        return None
    return tf(img).unsqueeze(0)


# ── Hook ──────────────────────────────────────────────────────────────────────
fmaps: Dict[str, torch.Tensor] = {}
def hook(_m, _in, out):
    fmaps["probe"] = out.detach()

h = probe_layer.register_forward_hook(hook)


# ── Indices ───────────────────────────────────────────────────────────────────
idx_pos = parse_idx_list(args.features_pos)
idx_neg = parse_idx_list(args.features_neg)
print(f"[info] features_pos={idx_pos} | features_neg={idx_neg}")

# ── Process & Compose 4×4 tile ────────────────────────────────────────────────
rows = cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
axes = np.asarray(axes).reshape(rows, cols)

valid_count = 0
for i in range(rows * cols):
    ax = axes[i // cols, i % cols]
    ax.axis("off")
    if i >= len(paths):
        continue

    pth = paths[i]
    xt = load_tensor(pth)
    if xt is None:
        continue
    xt = xt.to(device)

    with torch.no_grad():
        _ = net(xt)

    if "probe" not in fmaps:
        warnings.warn(f"[warn] no fmap captured for {pth.name}")
        continue

    fmap = fmaps["probe"].detach().cpu().squeeze(0)  # C×h×w
    C, h_, w_ = fmap.shape
    up = F.interpolate(fmap.unsqueeze(0), size=(args.img_size, args.img_size),
                       mode="bilinear", align_corners=False).squeeze(0)  # C×H×W
    up[up>1] = 1

    sel_pos = [k for k in idx_pos if 0 <= k < C]
    sel_neg = [k for k in idx_neg if 0 <= k < C]
    if len(sel_pos) < len(idx_pos) or len(sel_neg) < len(idx_neg):
        warnings.warn(f"[note] {pth.name}: some indices out of range (C={C}) were ignored")

    pos_map = up[sel_pos].sum(0) if sel_pos else torch.zeros_like(up[0])
    neg_map = up[sel_neg].sum(0) if sel_neg else torch.zeros_like(up[0])
    diff = pos_map - neg_map
    diff = minmax01(diff)
    diff[diff<0.5] = 0
    diff[diff>=0.5] = 1

    base = denorm_img(xt.cpu())
    ax.imshow(base, alpha=1.0)
    ax.imshow(diff.numpy(), cmap="magma", vmin=0, vmax=1, alpha=args.feat_alpha)
    ax.set_title(pth.name, fontsize=9)
    valid_count += 1

h.remove()

out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
fname = (args.outfile if args.outfile
         else f"choice_tile_{args.model}_{probe_path.replace('.', '_')}.png")
out_path = out_dir / fname
fig.suptitle(f"{args.model} | probe={probe_path} | Σpos−Σneg | pos={idx_pos} neg={idx_neg}", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out_path, dpi=180)
plt.close(fig)

h.remove()
print(f"[✓] tile saved → {out_path} (images used: {valid_count})")
