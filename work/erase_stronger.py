# erase_stronger.py
# Маскирование «жёлтых» областей теплокарт: режимы value / rank / alpha + дилатация (морф. расширение).
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError
from tqdm import tqdm

EPS = 1e-12

# ---------- чтение карты (0..1) ----------
def read_gray_arr01(stem: str, in_dir: Path, source: str = "auto"):
    """
    Возвращает (H,W) float32 в [0,1].
    source: 'npy' | 'png' | 'auto' (сначала npy, иначе png)
    """
    if source in ("auto", "npy"):
        npy = in_dir / f"{stem}.heat.npy"
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
        gray = in_dir / f"{stem}.heat.gray.png"
        if not gray.is_file():
            return None
        try:
            with Image.open(gray) as g: g = g.convert("L")
            return np.asarray(g, dtype=np.float32) / 255.0
        except (UnidentifiedImageError, OSError):
            return None
    return None

# ---------- пороговые правила ----------
def mask_value(arr01: np.ndarray, percent: float) -> np.ndarray:
    # как раньше: tau = max - (max - min)*(percent/100)
    vmin, vmax = float(arr01.min()), float(arr01.max())
    rng = vmax - vmin
    if rng <= EPS or percent <= 0: return np.zeros_like(arr01, bool)
    if percent >= 100: return np.ones_like(arr01, bool)
    tau = vmax - rng * (percent / 100.0)
    return arr01 >= tau

def mask_rank(arr01: np.ndarray, percent: float) -> np.ndarray:
    # квантиль: стераем примерно top-q% пикселей по рангу (гарантированная площадь)
    if percent <= 0: return np.zeros_like(arr01, bool)
    if percent >= 100: return np.ones_like(arr01, bool)
    q = 1.0 - (percent / 100.0)
    tau = float(np.quantile(arr01, q))
    return arr01 >= tau

def mask_alpha(arr01: np.ndarray, alpha: float) -> np.ndarray:
    # порог от пика: arr >= alpha*max(arr)
    vmax = float(arr01.max())
    if vmax <= EPS: return np.zeros_like(arr01, bool)
    alpha = float(alpha)
    alpha = 0.0 if alpha < 0 else (1.0 if alpha > 1.0 else alpha)
    tau = vmax * alpha
    return arr01 >= tau

# ---------- морфология ----------
def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0: return mask
    # через PIL MaxFilter; size = 2r+1
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    sz = max(1, 2 * radius + 1)
    im = im.filter(ImageFilter.MaxFilter(size=sz))
    out = (np.asarray(im, dtype=np.uint8) > 0)
    return out

# ---------- resize маски под PNG ----------
def resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    Ht, Wt = target_hw
    if mask.shape == (Ht, Wt): return mask
    im = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    im = im.resize((Wt, Ht), resample=Image.NEAREST)
    return (np.asarray(im, dtype=np.uint8) > 0)

# ---------- применение и сохранение ----------
def save_erased_png(src_path: Path, mask: np.ndarray, out_dir: Path, mode: str) -> bool:
    if not src_path.is_file(): return False
    try:
        with Image.open(src_path) as im_in:
            im = im_in.convert(mode)
        arr = np.array(im, dtype=np.uint8, copy=True)  # обязательно writeable
        Ht, Wt = arr.shape[:2]
        m = resize_mask(mask, (Ht, Wt))
        arr[m] = 0
        Image.fromarray(arr, mode=mode).save(out_dir / src_path.name)
        return True
    except (UnidentifiedImageError, OSError):
        return False

def main():
    p = argparse.ArgumentParser("Erase heatmaps by stronger rules (rank/alpha) + dilation.")
    p.add_argument("--heatmaps-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["value", "rank", "alpha"], default="rank",
                   help="value: верхние q% диапазона; rank: top-q%% по рангу; alpha: arr>=alpha*max")
    p.add_argument("--percent", type=float, default=20.0, help="q для value/rank (в %)")
    p.add_argument("--alpha", type=float, default=0.8, help="alpha для режима 'alpha' (0..1)")
    p.add_argument("--dilate", type=int, default=2, help="радиус дилатации маски (в пикселях теплокарты)")
    p.add_argument("--mask-source", choices=["auto", "png", "npy"], default="auto")
    args = p.parse_args()

    in_dir = Path(args.heatmaps-dir if hasattr(args, 'heatmaps-dir') else args.heatmaps_dir)  # защитa от тире
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # собрать stems
    stems = set()
    for pth in in_dir.glob("*.heat.npy"): stems.add(pth.stem.replace(".heat", ""))
    for pth in in_dir.glob("*.heat.gray.png"): stems.add(pth.name.replace(".heat.gray.png", ""))
    for pth in in_dir.glob("*.heat.magma.png"): stems.add(pth.name.replace(".heat.magma.png", ""))
    if not stems:
        raise SystemExit(f"[fatal] no heatmaps in {in_dir}")

    n_total=n_saved_gray=n_saved_magma=n_empty=n_no_src=0
    for stem in tqdm(sorted(stems), desc="erasing"):
        n_total += 1
        arr01 = read_gray_arr01(stem, in_dir, args.mask_source)
        if arr01 is None:
            n_no_src += 1
            continue

        if args.mode == "value":
            mask = mask_value(arr01, args.percent)
        elif args.mode == "rank":
            mask = mask_rank(arr01, args.percent)
        else:  # alpha
            mask = mask_alpha(arr01, args.alpha)

        if args.dilate > 0:
            mask = dilate_bool(mask, args.dilate)

        gray = in_dir / f"{stem}.heat.gray.png"
        magma = in_dir / f"{stem}.heat.magma.png"

        if not mask.any():
            # копируем исходники без изменений
            if gray.is_file(): Image.open(gray).save(out_dir / gray.name); n_saved_gray += 1
            if magma.is_file(): Image.open(magma).save(out_dir / magma.name); n_saved_magma += 1
            n_empty += 1
            continue

        if gray.is_file() and save_erased_png(gray, mask, out_dir, "L"): n_saved_gray += 1
        if magma.is_file() and save_erased_png(magma, mask, out_dir, "RGB"): n_saved_magma += 1

    print("\n[SUMMARY]")
    print(f"  stems total             : {n_total}")
    print(f"  saved gray PNG          : {n_saved_gray}")
    print(f"  saved magma PNG         : {n_saved_magma}")
    print(f"  empty masks (copied src): {n_empty}")
    print(f"  skipped, no src         : {n_no_src}")
    print(f"[✓] output: {out_dir}")

if __name__ == "__main__":
    main()
