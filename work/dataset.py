# dataset.py
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from PIL import Image, UnidentifiedImageError
except Exception as e:
    raise RuntimeError("Pillow is required (pip install pillow).") from e

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


@dataclass
class LoaderConfig:
    dataset_root: str
    pos: str  # "cat" | "dog"
    img_size: int = 256
    batch_size: int = 24
    num_workers: int = 0
    pin_memory: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Robust PIL loader (skips broken files)
# ──────────────────────────────────────────────────────────────────────────────

def _pil_loader_safe(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None


def _iter_images(dirpath: Path) -> Iterable[str]:
    if not dirpath.exists():
        return []
    for root, _, files in os.walk(dirpath):
        for f in files:
            if Path(f).suffix.lower() in ALLOWED_EXTS:
                yield str(Path(root) / f)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# Expected layout (per user description):
#   dataset_for_cat/
#       cat/train/*.jpg
#       cat/val/*.jpg
#       not_cat/train/*.jpg
#       not_cat/val/*.jpg
#   dataset_for_dog/
#       dog/train/*.jpg
#       dog/val/*.jpg
#       not_dog/train/*.jpg
#       not_dog/val/*.jpg
# ──────────────────────────────────────────────────────────────────────────────

class SafeBinaryDataset(Dataset):
    def __init__(self,
                 pos_files: List[str],
                 neg_files: List[str],
                 transform: Optional[Callable] = None):
        self.pos_files = pos_files
        self.neg_files = neg_files
        self.files = [(p, 1) for p in pos_files] + [(n, 0) for n in neg_files]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path, label = self.files[idx]
        img = _pil_loader_safe(path)
        if img is None:
            # Signal to collate_fn to drop this sample
            return None
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _split_dirs(root: Path, pos: str, split: str) -> Tuple[Path, Path]:
    neg = f"not_{pos}"
    pos_dir = root / pos / split
    neg_dir = root / neg / split
    return pos_dir, neg_dir


# ──────────────────────────────────────────────────────────────────────────────
# Transforms (prefer user's transform.py if present)
# ──────────────────────────────────────────────────────────────────────────────

def _build_transforms(img_size: int):
    try:
        from transform import build_transforms  # type: ignore
        return build_transforms(img_size)
    except Exception:
        # Minimal internal fallback
        from torchvision import transforms as T
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        val_tf = T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return train_tf, val_tf


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def build_datasets(dataset_root: str, pos: str, img_size: int) -> Tuple[Dataset, Dataset]:
    root = Path(dataset_root)
    train_pos_dir, train_neg_dir = _split_dirs(root, pos, "train")
    val_pos_dir, val_neg_dir = _split_dirs(root, pos, "val")

    train_tf, val_tf = _build_transforms(img_size)

    train_ds = SafeBinaryDataset(
        pos_files=list(_iter_images(train_pos_dir)),
        neg_files=list(_iter_images(train_neg_dir)),
        transform=train_tf
    )
    val_ds = SafeBinaryDataset(
        pos_files=list(_iter_images(val_pos_dir)),
        neg_files=list(_iter_images(val_neg_dir)),
        transform=val_tf
    )
    return train_ds, val_ds


def build_dataloaders(dataset_root: str,
                      pos: str,
                      img_size: int,
                      batch_size: int,
                      num_workers: int = 0,
                      pin_memory: bool = False) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = build_datasets(dataset_root=dataset_root, pos=pos, img_size=img_size)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_skip_none,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_skip_none,
    )
    return train_dl, val_dl


def build_val_loader(dataset_root: str,
                     pos: str,
                     img_size: int,
                     batch_size: int,
                     num_workers: int = 0,
                     pin_memory: bool = False) -> DataLoader:
    """
    Convenience for evaluate.py: returns only the validation loader.
    """
    _, val_ds = build_datasets(dataset_root=dataset_root, pos=pos, img_size=img_size)
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_skip_none,
    )
    return val_dl
