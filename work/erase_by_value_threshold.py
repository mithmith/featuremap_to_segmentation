# erase_by_value_threshold_v2.py
# Зануление верхних q% диапазона значений теплокарты: tau = max - (max - min) * (q/100).
# Робастно обрабатывает несовпадение размеров маски и PNG, пишет подробную статистику.
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

EPS = 1e-12

def read_gray_arr01(stem: str, in_dir: Path, source: str = "auto"):
    """
    Возвращает (H,W) float32 в [0,1] для теплокарты.
    source: 'npy' | 'png' | 'auto' (по умолчанию 'auto': сперва npy, иначе png)
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
            with Image.open(gray) as g_im:
                g = g_im.convert("L")
            arr = np.asarray(g, dtype=np.float32) / 255.0
            return arr
        except (UnidentifiedImageError, OSError):
            return None

    return None

def value_threshold_mask(arr01: np.ndarray, percent: float) -> np.ndarray:
    vmin = float(arr01.min())
    vmax = float(arr01.max())
    rng  = vmax - vmin
    if rng <= EPS or percent <= 0:
        return np.zeros_like(arr01, dtype=bool)
    if percent >= 100:
        return np.ones_like(arr01, dtype=bool)
    tau = vmax - rng * (percent / 100.0)
    return arr01 >= tau

def resize_mask(mask: np.ndarray, target_hw: tuple[int,int]) -> np.ndarray:
    """Масштабирование bool-маски до нужного H×W (nearest)."""
    Ht, Wt = target_hw
    if mask.shape == (Ht, Wt):
        return mask
    # PIL ожидает (W,H)
    im = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    im = im.resize((Wt, Ht), resample=Image.NEAREST)
    out = (np.asarray(im, dtype=np.uint8) > 0)
    return out

def save_erased_png(src_path: Path, mask: np.ndarray, out_dir: Path, mode: str) -> bool:
    """
    Стирает пиксели по маске и сохраняет PNG.
    mode: 'L' для gray, 'RGB' для magma.
    Возвращает True, если сохранено.
    """
    if not src_path.is_file():
        return False
    try:
        with Image.open(src_path) as im_in:
            im = im_in.convert(mode)
        arr = np.array(im, dtype=np.uint8, copy=True)  # writeable
        # Подгоняем маску под размер изображения
        Ht, Wt = arr.shape[:2]
        m = resize_mask(mask, (Ht, Wt))
        if mode == "L":
            arr[m] = 0
            Image.fromarray(arr, mode=mode).save(out_dir / src_path.name)
        else:  # RGB
            arr[m] = 0
            Image.fromarray(arr, mode=mode).save(out_dir / src_path.name)
        return True
    except (UnidentifiedImageError, OSError):
        return False

def main():
    ap = argparse.ArgumentParser("Erase by value-threshold (upper q% of intensity range) with robust saving.")
    ap.add_argument("--heatmaps-dir", required=True, help="Папка с *.heat.npy/png")
    ap.add_argument("--out-dir", required=True, help="Куда сохранять обрезанные PNG")
    ap.add_argument("--percent", type=float, default=10.0, help="Процент верхней части диапазона яркостей")
    ap.add_argument("--mask-source", choices=["auto", "png", "npy"], default="auto",
                    help="Источник значений для порога (по умолчанию auto: npy→png)")
    args = ap.parse_args()

    in_dir  = Path(args.heatmaps_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # собираем stem'ы
    stems = set()
    for p in in_dir.glob("*.heat.npy"):
        stems.add(p.stem.replace(".heat", ""))
    for p in in_dir.glob("*.heat.gray.png"):
        stems.add(p.name.replace(".heat.gray.png", ""))
    for p in in_dir.glob("*.heat.magma.png"):
        stems.add(p.name.replace(".heat.magma.png", ""))

    if not stems:
        raise SystemExit(f"[fatal] no heatmaps found in: {in_dir}")

    n_total = 0
    n_mask_empty = 0
    n_saved_gray = 0
    n_saved_magma = 0
    n_skipped_no_src = 0

    for stem in tqdm(sorted(stems), desc="erasing (value-threshold)"):
        n_total += 1
        arr01 = read_gray_arr01(stem, in_dir, args.mask_source)
        if arr01 is None:
            n_skipped_no_src += 1
            continue

        mask = value_threshold_mask(arr01, args.percent)
        gray_path  = in_dir / f"{stem}.heat.gray.png"
        magma_path = in_dir / f"{stem}.heat.magma.png"

        if not mask.any():
            # маска пустая — просто копируем исходные файлы как есть
            if gray_path.is_file():
                with Image.open(gray_path) as im: im.save(out_dir / gray_path.name)
                n_saved_gray += 1
            if magma_path.is_file():
                with Image.open(magma_path) as im: im.save(out_dir / magma_path.name)
                n_saved_magma += 1
            n_mask_empty += 1
            continue

        # применяем маску
        if gray_path.is_file():
            if save_erased_png(gray_path, mask, out_dir, mode="L"):
                n_saved_gray += 1
        if magma_path.is_file():
            if save_erased_png(magma_path, mask, out_dir, mode="RGB"):
                n_saved_magma += 1

    print("\n[SUMMARY]")
    print(f"  stems total       : {n_total}")
    print(f"  saved gray PNG    : {n_saved_gray}")
    print(f"  saved magma PNG   : {n_saved_magma}")
    print(f"  empty masks (copied orig) : {n_mask_empty}")
    print(f"  skipped (no npy/png for mask source): {n_skipped_no_src}")
    print(f"[✓] output: {out_dir}")

if __name__ == "__main__":
    main()
