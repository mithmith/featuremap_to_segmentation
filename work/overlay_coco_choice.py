# overlay_coco_choice.py
# Наложение: ОРИГИНАЛ + TRIMAP + Σ(фич) (либо Σpos−Σneg) в один PNG на картинку.
# Совместимо по интерфейсу с choice_features.py (модель, веса, probe, индексы каналов).

import argparse, importlib, inspect, warnings
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
ap = argparse.ArgumentParser("Overlay: image + COCO/Oxford trimap + Σfeatures (or Σpos−Σneg)")
ap.add_argument("--model", required=True, help="Модуль с классом Model/CNNModel (alexnet, resnet, ...)")
ap.add_argument("--weights", required=True, help="Путь к .pt (state_dict или чекпойнт)")
ap.add_argument("--images-dir", required=True, help=r"Каталог с изображениями")
ap.add_argument("--masks-dir", required=True, help=r"Каталог с trimaps (например ...\annotations\trimaps)")
ap.add_argument("--probe", required=True, help="Слой-хук: 'before_pool3' или путь 'features[8]'/'block4.bn2' и т.п.")
ap.add_argument("--img-size", type=int, default=224)
ap.add_argument("--device", choices=["auto", "cpu", "cuda", "dml"], default="auto")

ap.add_argument("--features-pos", nargs="*", default=[], help="Индексы положит. каналов (напр. 6 11 12 или [6,11,12])")
ap.add_argument("--features-neg", nargs="*", default=[], help="Индексы отрицат. каналов (напр. 25 29)")
ap.add_argument("--mode", choices=["sum", "diff"], default="sum", help="sum=Σpos; diff=Σpos−Σneg (как в choice_features.py)")
ap.add_argument("--samples", type=int, default=0, help="0=все; иначе ограничить количеством файлов")
ap.add_argument("--out", default=r"D:\progs\work\features_project\overlays", help="Куда сохранять PNG")
ap.add_argument("--outfile-suffix", default="", help="Суффикс в имени файла (по желанию)")

ap.add_argument("--orig-alpha", type=float, default=1.0, help="Прозрачность базового изображения")
ap.add_argument("--mask-alpha", type=float, default=0.35, help="Прозрачность тримап-маски")
ap.add_argument("--feat-alpha", type=float, default=0.50, help="Прозрачность карты признаков")
ap.add_argument("--feat-cmap", default="magma", help="Колормап для карты признаков (magma/jet/…)")
ap.add_argument("--mask-cmap", default="Reds", help="Колормап для маски (Reds/Blues/Greens/…)")
ap.add_argument("--trimap-fg", choices=["2", "23"], default="23", help="Какие метки считать объектом: '2' (строго fg) или '23' (fg+border)")

ap.add_argument("--title-fs", type=float, default=7.0, help="Размер шрифта заголовка (по умолчанию 7)")
ap.add_argument("--tight-top", type=float, default=0.97, help="Отступ сверху для tight_layout [0..1] (по умолчанию 0.97)")


args = ap.parse_args()

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
            print(f"[warn] torch-directml не доступен ({e}); CPU")
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
        raise ValueError(f"'{path}' не nn.Module")
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
    # x: 1×3×H×W, нормирован под mean=0.5 std=0.5
    return (x.squeeze(0).mul(0.5).add(0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy())

# ── I/O ───────────────────────────────────────────────────────────────────────
img_dir = Path(args.images-dir if hasattr(args, "images-dir") else args.images_dir)  # безопасно на случай дефиса
img_dir = Path(args.images_dir)
mask_dir = Path(args.masks_dir)
if not img_dir.is_dir():
    raise SystemExit(f"[fatal] images-dir не найден: {img_dir}")
if not mask_dir.is_dir():
    raise SystemExit(f"[fatal] masks-dir не найден: {mask_dir}")

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
paths = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts and p.is_file()]
if args.samples and args.samples > 0:
    paths = paths[: args.samples]
if not paths:
    raise SystemExit(f"[fatal] нет изображений в {img_dir}")

# трансформ для входа в сеть (как в choice_features.py)
tf_img = T.Compose([
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
    return tf_img(img).unsqueeze(0)

def load_trimap_mask(stem: str) -> np.ndarray | None:
    """
    Возвращает H×W float64 в {0,1} после ресайза к (img_size,img_size).
    Ищем файл по имени stem.* в masks-dir (обычно .png).
    Oxford trimap: значения в {1,2,3}. '2' — объект, '3' — граница. По умолчанию берём {2,3}.
    """
    # кандидаты имён
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        cand = mask_dir / f"{stem}{ext}"
        if cand.is_file():
            try:
                m = Image.open(cand)
            except (UnidentifiedImageError, OSError):
                return None
            # чтобы не расползалась маска — NEAREST
            m = m.resize((args.img_size, args.img_size), resample=Image.NEAREST)
            # в numpy
            mnp = np.array(m)
            # если палитра/цвет — переведём к ints
            if mnp.ndim == 3:
                # возьмём один канал (маски часто в палитре), либо сделаем яркостный
                mnp = mnp[..., 0]
            vals = set(np.unique(mnp).tolist())
            # бинализация по тримап
            if args.trimap_fg == "2":
                mask = (mnp == 2)
            else:  # "23"
                mask = np.logical_or(mnp == 2, mnp == 3)
            return mask.astype(np.float64)
    return None

# ── Model & hook ──────────────────────────────────────────────────────────────
mod = importlib.import_module(args.model)
Model = resolve_model_class(mod)
net = Model().to(device).eval()

state = torch.load(args.weights, map_location="cpu")
if isinstance(state, dict) and "model" in state:
    state = state["model"]
# снять 'module.' при необходимости
if isinstance(state, dict):
    state = { (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v for k, v in state.items() }
net.load_state_dict(state, strict=False)

if args.probe.startswith("before_pool"):
    mapping = find_convs_before_pools(net)
    if args.probe not in mapping:
        avail = ", ".join(mapping.keys()) if mapping else "<none>"
        raise SystemExit(f"[fatal] probe '{args.probe}' не найден. Доступно: {avail}")
    probe_path, probe_layer = mapping[args.probe]
else:
    probe_path, probe_layer = args.probe, get_module_by_path(net, args.probe)

print(f"[info] device={device} | probe={args.probe} -> {probe_path}")

fmaps: Dict[str, torch.Tensor] = {}
def hook(_m, _in, out):
    fmaps["probe"] = out.detach()
h = probe_layer.register_forward_hook(hook)

idx_pos = parse_idx_list(args.features_pos)
idx_neg = parse_idx_list(args.features_neg)
print(f"[info] features_pos={idx_pos} | features_neg={idx_neg} | mode={args.mode}")

out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

# ── Main loop ─────────────────────────────────────────────────────────────────
saved = 0
for p in paths:
    stem = p.stem
    xt = load_tensor(p)
    if xt is None:
        continue
    xt = xt.to(device)

    # маска
    mask = load_trimap_mask(stem)
    if mask is None:
        warnings.warn(f"[skip] нет подходящей маски для {stem}")
        continue

    with torch.no_grad():
        _ = net(xt)

    if "probe" not in fmaps:
        warnings.warn(f"[warn] нет fmap для {p.name}")
        continue

    fmap = fmaps["probe"].detach().cpu().squeeze(0)  # C×h×w
    C, _, _ = fmap.shape
    up = F.interpolate(fmap.unsqueeze(0), size=(args.img_size, args.img_size),
                       mode="bilinear", align_corners=False).squeeze(0)  # C×H×W

    # выбор каналов
    sel_pos = [k for k in idx_pos if 0 <= k < C]
    sel_neg = [k for k in idx_neg if 0 <= k < C]
    if len(sel_pos) < len(idx_pos) or len(sel_neg) < len(idx_neg):
        warnings.warn(f"[note] {p.name}: часть индексов вне диапазона (C={C}) — пропущены")

    # Σ карт
    pos_map = up[sel_pos].sum(0) if sel_pos else torch.zeros_like(up[0])
    if args.mode == "sum":
        comp = pos_map
    else:
        neg_map = up[sel_neg].sum(0) if sel_neg else torch.zeros_like(up[0])
        comp = pos_map - neg_map
    comp = minmax01(comp).cpu().numpy()  # H×W в [0,1]

    # база и визуализация
    base = denorm_img(xt.cpu())  # H×W×3 (0..1)
    mask01 = (mask > 0).astype(np.float64)  # H×W {0,1}

    # отрисовка: всё в один кадр
    fig = plt.figure(figsize=(4.8, 4.8), dpi=180)
    ax = fig.add_subplot(1,1,1)
    ax.imshow(base, alpha=args.orig_alpha)  # базовое изображение
    ax.imshow(mask01, cmap=args.mask_cmap, vmin=0, vmax=1, alpha=args.mask_alpha)  # тримап
    ax.imshow(comp, cmap=args.feat_cmap, vmin=0, vmax=1, alpha=args.feat_alpha)    # карта признаков
    ax.set_axis_off()
    title = f"{stem} | {args.model} @ {probe_path} | pos={sel_pos}" + (f" neg={sel_neg}" if args.mode=="diff" and sel_neg else "")
    ax.set_title(title, fontsize=args.title_fs)
    plt.tight_layout(rect=[0, 0, 1, args.tight_top])


    suffix = f"_{args.outfile_suffix}" if args.outfile_suffix else ""
    mode_tag = "sum" if args.mode == "sum" else "diff"
    out_path = out_dir / f"{stem}__overlay_{args.model}_{mode_tag}{suffix}.png"
    fig.savefig(out_path)
    plt.close(fig)
    saved += 1

h.remove()
print(f"[✓] overlays saved: {saved} → {out_dir}")
