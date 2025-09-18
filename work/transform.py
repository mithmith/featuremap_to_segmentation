# transform.py
from dataclasses import dataclass
from typing import Tuple, Optional, Union

try:
    from torchvision import transforms as T
except Exception as e:
    raise RuntimeError(
        "torchvision is required for transforms. Please install torchvision "
        "(pip install torchvision) and try again."
    ) from e


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class AugConfig:
    img_size: int = 256
    # RandomResizedCrop params
    scale: Tuple[float, float] = (0.7, 1.0)        # keep at least 70% of the image
    ratio: Tuple[float, float] = (0.8, 1.25)       # allow moderate aspect shifts
    # Geometric augs
    rotate_deg: float = 15.0
    hflip_p: float = 0.5
    # Color augs (disabled by default; enable if needed)
    color_jitter: Optional[Tuple[float, float, float, float]] = None  # (b, c, s, h)
    # Normalization
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std: Tuple[float, float, float] = IMAGENET_STD
    # Resize factor for val (Resize -> CenterCrop)
    val_resize_factor: float = 1.14


def _train_transform(cfg: AugConfig) -> T.Compose:
    ops = [
        T.RandomResizedCrop(cfg.img_size, scale=cfg.scale, ratio=cfg.ratio),
        T.RandomHorizontalFlip(p=cfg.hflip_p),
        T.RandomRotation(degrees=cfg.rotate_deg),
    ]
    if cfg.color_jitter is not None:
        b, c, s, h = cfg.color_jitter
        ops.append(T.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h))
    ops += [
        T.ToTensor(),
        T.Normalize(cfg.mean, cfg.std),
    ]
    return T.Compose(ops)


def _val_transform(cfg: AugConfig) -> T.Compose:
    return T.Compose([
        T.Resize(int(cfg.img_size * cfg.val_resize_factor)),
        T.CenterCrop(cfg.img_size),
        T.ToTensor(),
        T.Normalize(cfg.mean, cfg.std),
    ])


def build_transforms(arg: Union[int, AugConfig]) -> Tuple[T.Compose, T.Compose]:
    """
    Returns (train_tf, val_tf).

    Accepts either:
      - img_size: int
      - AugConfig
    """
    cfg = AugConfig(img_size=arg) if isinstance(arg, int) else arg
    return _train_transform(cfg), _val_transform(cfg)


def build_val_transform(arg: Union[int, AugConfig]) -> T.Compose:
    cfg = AugConfig(img_size=arg) if isinstance(arg, int) else arg
    return _val_transform(cfg)
