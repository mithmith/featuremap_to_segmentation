# Применяет маску стирания (из теплокарт) к исходным изображениям и сохраняет "чёрные прогалы".
import argparse
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError
from tqdm import tqdm

EPS = 1e-12
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---------- чтение карты (0..1) ----------
def read_heat_arr01(stem: str, heatmaps_dir: Path, source: str = "auto"):
    """
    Возвращает (H,W) float32 в [0,1] из *.heat.npy или *.heat.gray.png.
    source: 'npy' | 'png' | 'auto'
    """
    if source in ("auto", "npy"):
        npy = heatmaps_dir / f"{stem}.heat.npy"
        if npy.is_file():
            try:
                arr = np.load(npy).astype(np.float32)
                vmin, vmax = float(arr.min()), float(arr.max())
                if vmax - vmin > EPS:
                    arr = (arr - vmin) / (vmax - vmin)
                else:
                    arr = np.zeros_like(arr, dtype=np.float32)
                return arr
            except Exception:
                if source == "npy":
                    return None
    if source in ("auto", "png"):
        gray = heatmaps_dir / f"{stem}.heat.gray.png"
        if not gray.is_file():
            return None
        try:
            with Image.open(gray) as g_im:
                g = g_im.convert("L")
            return np.asarray(g, dtype=np.float32) / 255.0
        except (UnidentifiedImageError, OSError):
            return None
    return None

# ---------- правила порогования ----------
def mask_value(arr01: np.ndarray, percent: float) -> np.ndarray:
    vmin, vmax = float(arr01.min()), float(arr01.max())
    rng = vmax - vmin
    if rng <= EPS or percent <= 0: return np.zeros_like(arr01, bool)
    if percent >= 100: return np.ones_like(arr01, bool)
    tau = vmax - rng * (percent / 100.0)
    return arr01 >= tau

def mask_rank(arr01: np.ndarray, percent: float) -> np.ndarray:
    if percent <= 0: return np.zeros_like(arr01, bool)
    if percent >= 100: return np.ones_like(arr01, bool)
    q = 1.0 - (percent / 100.0)
    tau = float(np.quantile(arr01, q))
    return arr01 >= tau

def mask_alpha(arr01: np.ndarray, alpha: float) -> np.ndarray:
    vmax = float(arr01.max())
    if vmax <= EPS: return np.zeros_like(arr01, bool)
    alpha = max(0.0, min(1.0, float(alpha)))
    tau = vmax * alpha
    return arr01 >= tau

def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0: return mask
    im = Image.fromarray(mask.astype(np.uint8)*255, mode="L")
    sz = max(1, 2*radius + 1)
    im = im.filter(ImageFilter.MaxFilter(size=sz))
    return (np.asarray(im, dtype=np.uint8) > 0)

def resize_mask(mask: np.ndarray, target_hw: tuple[int,int]) -> np.ndarray:
    Ht, Wt = target_hw
    if mask.shape == (Ht, Wt): return mask
    im = Image.fromarray(mask.astype(np.uint8)*255, mode="L")
    im = im.resize((Wt, Ht), resample=Image.NEAREST)
    return (np.asarray(im, dtype=np.uint8) > 0)

# ---------- геометрия под вход сети (Resize→CenterCrop) ----------
def resize_shorter_keep_aspect(pil: Image.Image, shorter: int) -> Image.Image:
    w, h = pil.size
    if w == 0 or h == 0: return pil
    if w < h:
        new_w = shorter
        new_h = int(round(h * (shorter / w)))
    else:
        new_h = shorter
        new_w = int(round(w * (shorter / h)))
    return pil.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

def center_crop(pil: Image.Image, size: int) -> Image.Image:
    w, h = pil.size
    th = tw = size
    if w == tw and h == th: return pil
    i = max(0, (h - th) // 2)
    j = max(0, (w - tw) // 2)
    return pil.crop((j, i, j + tw, i + th))

def geom_align_like_val(pil: Image.Image, img_size: int, resize_factor: float = 1.14) -> Image.Image:
    shorter = int(round(img_size * resize_factor))
    x = resize_shorter_keep_aspect(pil, shorter)
    x = center_crop(x, img_size)
    return x

# ---------- применение маски к изображению ----------
def apply_mask_to_image(pil_rgb: Image.Image, mask: np.ndarray, black_value: int = 0) -> Image.Image:
    """
    pil_rgb: PIL RGB (HxW)
    mask: bool (H×W)
    """
    arr = np.array(pil_rgb, dtype=np.uint8, copy=True)
    Ht, Wt = arr.shape[:2]
    m = resize_mask(mask, (Ht, Wt))
    arr[m] = black_value  # (0,0,0)
    return Image.fromarray(arr, mode="RGB")

def main():
    ap = argparse.ArgumentParser("Apply erasing mask (from heatmaps) to original photos.")
    ap.add_argument("--images-dir", required=True, help="Папка с исходными изображениями (cat/train).")
    ap.add_argument("--heatmaps-dir", required=True, help="Папка с теплокартами (*.heat.npy/png).")
    ap.add_argument("--out-dir", required=True, help="Куда сохранять фото с чёрными прогалами.")
    ap.add_argument("--mode", choices=["value", "rank", "alpha"], default="rank")
    ap.add_argument("--percent", type=float, default=20.0, help="Процент для value/rank.")
    ap.add_argument("--alpha", type=float, default=0.8, help="alpha для режима 'alpha' (0..1).")
    ap.add_argument("--dilate", type=int, default=2, help="радиус дилатации маски в пикселях теплокарты.")
    ap.add_argument("--mask-source", choices=["auto", "png", "npy"], default="auto")
    ap.add_argument("--img-size", type=int, default=256, help="целевой размер входа (должен совпадать с шагом 1).")
    ap.add_argument("--resize-factor", type=float, default=1.14, help="коэффициент для Resize(shorter_side=f·img_size).")
    ap.add_argument("--copy-if-missing", action="store_true",
                    help="Если нет теплокарты для файла — копировать оригинал в out-dir.")
    ap.add_argument("--black", type=int, default=0, help="значение чёрного (0..255), по умолчанию 0.")
    args = ap.parse_args()

    img_dir  = Path(args.images_dir)
    heat_dir = Path(args.heatmaps_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # индексация доступных heatmap-стемов
    heat_stems = set()
    for p in heat_dir.glob("*.heat.npy"): heat_stems.add(p.stem.replace(".heat", ""))
    for p in heat_dir.glob("*.heat.gray.png"): heat_stems.add(p.name.replace(".heat.gray.png", ""))
    for p in heat_dir.glob("*.heat.magma.png"): heat_stems.add(p.name.replace(".heat.magma.png", ""))

    if not heat_stems:
        raise SystemExit(f"[fatal] no heatmaps found in: {heat_dir}")

    imgs = [p for p in sorted(img_dir.rglob("*")) if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
    if not imgs:
        raise SystemExit(f"[fatal] no images found in: {img_dir}")

    n_total = n_saved = n_copied = n_skipped = 0

    for p in tqdm(imgs, desc="apply-mask"):
        n_total += 1
        stem = p.stem
        if stem not in heat_stems:
            if args.copy_if_missing:
                shutil.copy2(p, out_dir / p.name)
                n_copied += 1
            else:
                n_skipped += 1
            continue

        # читаем карту
        arr01 = read_heat_arr01(stem, heat_dir, source=args.mask_source)
        if arr01 is None:
            if args.copy_if_missing:
                shutil.copy2(p, out_dir / p.name); n_copied += 1
            else:
                n_skipped += 1
            continue

        # строим маску
        if args.mode == "value":
            mask = mask_value(arr01, args.percent)
        elif args.mode == "rank":
            mask = mask_rank(arr01, args.percent)
        else:
            mask = mask_alpha(arr01, args.alpha)

        if args.dilate > 0:
            mask = dilate_bool(mask, args.dilate)

        # грузим исходное изображение и приводим к входной геометрии
        try:
            with Image.open(p) as im_in:
                im = im_in.convert("RGB")
        except (UnidentifiedImageError, OSError):
            n_skipped += 1
            continue

        im_aligned = geom_align_like_val(im, img_size=args.img_size, resize_factor=args.resize_factor)

        # применяем маску и сохраняем
        erased = apply_mask_to_image(im_aligned, mask, black_value=args.black)
        out_path = out_dir / p.name  # сохраняем под тем же именем и расширением
        erased.save(out_path)
        n_saved += 1

    print("\n[SUMMARY]")
    print(f"  images total : {n_total}")
    print(f"  saved erased : {n_saved}")
    print(f"  copied orig  : {n_copied}")
    print(f"  skipped      : {n_skipped}")
    print(f"[✓] output: {out_dir}")

if __name__ == "__main__":
    main()
