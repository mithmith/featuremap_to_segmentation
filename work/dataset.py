from pathlib import Path
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List

# ---- helpers ----
def _img_files_in_dir(d: Path) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [str(p) for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def _detect_layout(root: Path) -> str:
    # split-first: root/train/<class>, root/val/<class>
    if (root / "train").exists() and (root / "val").exists():
        return "split_first"
    # class-first: root/<class>/{train,val}
    return "class_first"

def _gather_files(root: Path, split: str, pos: str, neg: str, layout: str) -> Tuple[List[str], List[str]]:
    if layout == "split_first":
        pos_dir = root / split / pos
        neg_dir = root / split / neg
    else:
        pos_dir = root / pos / split
        neg_dir = root / neg / split
    if not pos_dir.exists():
        raise FileNotFoundError(f"Not found: {pos_dir}")
    if not neg_dir.exists():
        raise FileNotFoundError(f"Not found: {neg_dir}")
    return _img_files_in_dir(pos_dir), _img_files_in_dir(neg_dir)

# ---- main API (совместимая версия) ----
def build_datasets(dataset_root: str, pos: str, img_size: int):
    """
    Поддерживает обе раскладки:
      A) split-first:   root/train/{animal,not_animal}, root/val/{animal,not_animal}
      B) class-first:   root/{animal,not_animal}/{train,val}
    """
    # трансформы берём как раньше (из transform.py), при ошибке — лёгкий фолбек
    try:
        from transform import build_transforms
        train_tf, val_tf = build_transforms(img_size)
    except Exception:
        # упрощённый фолбек без torchvision
        import numpy as np
        from PIL import Image
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        def _to_tensor(img: Image.Image):
            a = np.asarray(img, dtype=np.float32)
            if a.ndim == 2:
                a = np.stack([a, a, a], axis=-1)
            a = a / 255.0
            a = np.transpose(a, (2, 0, 1))
            return torch.from_numpy(a)
        class _T:
            def __init__(self, size): self.size = size
            def __call__(self, img: Image.Image):
                img = img.resize((self.size, self.size), Image.BILINEAR)
                x = _to_tensor(img)
                return (x - mean) / std
        train_tf, val_tf = _T(img_size), _T(img_size)

    root = Path(dataset_root)
    neg = "not_animal" if pos == "animal" else "animal"
    layout = _detect_layout(root)

    tr_pos, tr_neg = _gather_files(root, "train", pos, neg, layout)
    va_pos, va_neg = _gather_files(root, "val",   pos, neg, layout)

    train_ds = SafeBinaryDataset(tr_pos, tr_neg, transform=train_tf)
    val_ds   = SafeBinaryDataset(va_pos, va_neg, transform=val_tf)
    return train_ds, val_ds


def build_dataloaders(dataset_root: str,
                      pos: str,
                      img_size: int,
                      batch_size: int,
                      num_workers: int = 0,
                      pin_memory: bool = False) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = build_datasets(dataset_root, pos, img_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory,
                          collate_fn=_collate_skip_none)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory,
                          collate_fn=_collate_skip_none)
    return train_dl, val_dl


def build_val_loader(dataset_root: str,
                     pos: str,
                     img_size: int,
                     batch_size: int,
                     num_workers: int = 0,
                     pin_memory: bool = False) -> DataLoader:
    _, val_ds = build_datasets(dataset_root, pos, img_size)
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory,
                      collate_fn=_collate_skip_none)
