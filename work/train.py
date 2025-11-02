# train.py — быстрый DML-safe тренер под структуру root/train|val/{animal,not_animal}
import argparse
import csv
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageEnhance, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ───────────────────────── Device ─────────────────────────
def get_device(name: str) -> torch.device:
    name = name.lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "dml":
        try:
            import torch_directml  # noqa: F401
            return torch_directml.device()
        except Exception as e:
            tqdm.write(f"[warn] torch-directml not available ({e}); falling back to CPU.")
    return torch.device("cpu")


# ─────────────── Аугментации (без torchvision) ───────────────
class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, img): 
        for t in self.transforms: img = t(img)
        return img

class Resize:
    def __init__(self, size): self.size = size
    def __call__(self, img: Image.Image):
        return img.resize((self.size, self.size), Image.BICUBIC)

class CenterCrop:
    def __init__(self, size): self.size = size
    def __call__(self, img: Image.Image):
        w, h = img.size; s = self.size
        i = max(0, (h - s) // 2); j = max(0, (w - s) // 2)
        return img.crop((j, i, j + s, i + s))

class RandomHorizontalFlip:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img: Image.Image):
        return ImageOps.mirror(img) if random.random() < self.p else img

class RandomRotation:
    def __init__(self, degrees=15): self.degrees = degrees
    def __call__(self, img: Image.Image):
        ang = random.uniform(-self.degrees, self.degrees)
        return img.rotate(ang, resample=Image.BICUBIC, expand=False, fillcolor=(0,0,0))

class RandomResizedCrop:
    def __init__(self, size, scale=(0.7, 1.0), ratio=(0.8, 1.25)):
        self.size = size; self.scale = scale; self.ratio = ratio
    def __call__(self, img: Image.Image):
        w, h = img.size; area = w * h
        for _ in range(10):
            ta = random.uniform(*self.scale) * area
            ar = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            nw, nh = int(round((ta * ar) ** 0.5)), int(round((ta / ar) ** 0.5))
            if 0 < nw <= w and 0 < nh <= h:
                i = random.randint(0, h - nh); j = random.randint(0, w - nw)
                return img.crop((j, i, j + nw, i + nh)).resize((self.size, self.size), Image.BICUBIC)
        s = min(w, h); i = (h - s) // 2; j = (w - s) // 2
        return img.crop((j, i, j + s, i + s)).resize((self.size, self.size), Image.BICUBIC)
    
    # ─────────────── Случайное зануление прямоугольника ───────────────
class RandomZeroRect:
    def __init__(self, p: float = 0.1, min_size: int = 4, max_size: int = 32):
        assert 0.0 <= p <= 1.0
        assert 1 <= min_size <= max_size
        self.p = p
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: Image.Image) -> Image.Image:
        # ожидание RGB (в ваших лоадерах .convert("RGB") уже сделан)
        if random.random() >= self.p:
            return img

        w, h = img.size
        if w < self.min_size or h < self.min_size:
            return img

        rw = random.randint(self.min_size, min(self.max_size, w))
        rh = random.randint(self.min_size, min(self.max_size, h))
        # координаты левого верхнего угла
        x0 = random.randint(0, w - rw)
        y0 = random.randint(0, h - rh)

        # чёрный патч нужного размера
        patch = Image.new("RGB", (rw, rh), (0, 0, 0))
        img.paste(patch, (x0, y0))
        return img

class _GaussNoisePIL:
    def __init__(self, sigma_range: Tuple[float, float] = (5.0, 30.0)):
        self.sigma_range = sigma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img, dtype=np.float32)
        sigma = random.uniform(*self.sigma_range)
        noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

class _SaltPepperPIL:
    def __init__(self, amount_range: Tuple[float, float] = (0.01, 0.05), salt_ratio: float = 0.5):
        self.amount_range = amount_range
        self.salt_ratio = salt_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img, dtype=np.uint8)
        h, w, _ = arr.shape
        amount = random.uniform(*self.amount_range)  # доля пикселей (0..1)
        m = np.random.rand(h, w)
        salt = m < (amount * self.salt_ratio)
        pepper = (m >= (amount * self.salt_ratio)) & (m < amount)
        out = arr.copy()
        out[salt] = 255
        out[pepper] = 0
        return Image.fromarray(out, mode="RGB")

class RandomNoiseOneOf:
    """
    Аналог Albumentations: OneOf([GaussNoise, Salt&Pepper], p=0.25).
    Выполняется на PIL → размещать до ToTensor().
    """
    def __init__(
        self,
        p: float = 0.25,
        gaussian_sigma: Tuple[float, float] = (5.0, 30.0),
        sp_amount: Tuple[float, float] = (0.01, 0.05),
        sp_salt_ratio: float = 0.5,
    ):
        assert 0.0 <= p <= 1.0
        self.p = p
        self.gn = _GaussNoisePIL(gaussian_sigma)
        self.sp = _SaltPepperPIL(sp_amount, salt_ratio=sp_salt_ratio)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        return self.gn(img) if random.random() < 0.5 else self.sp(img)
    
    # ─────────────── ColorJitter / ToGray (OneOf) ───────────────
def _shift_hue(img: Image.Image, delta_frac: float) -> Image.Image:
    # delta_frac ∈ [-0.5, 0.5]; PIL HSV: H ∈ [0,255]
    hsv = np.array(img.convert("HSV"), dtype=np.uint8)
    h = hsv[..., 0].astype(np.int16)
    shift = int(round(delta_frac * 255.0))
    h = (h + shift) % 256
    hsv[..., 0] = h.astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")

class ColorJitterPIL:
    """Яркость/контраст/насыщенность/тон как в torchvision."""
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        self.b = float(brightness); self.c = float(contrast)
        self.s = float(saturation); self.h = float(hue)

    def __call__(self, img: Image.Image) -> Image.Image:
        ops = []
        if self.b > 0: ops.append(("b", 1.0 + random.uniform(-self.b, self.b)))
        if self.c > 0: ops.append(("c", 1.0 + random.uniform(-self.c, self.c)))
        if self.s > 0: ops.append(("s", 1.0 + random.uniform(-self.s, self.s)))
        if self.h > 0: ops.append(("h", random.uniform(-self.h, self.h)))  # frac

        random.shuffle(ops)
        for kind, v in ops:
            if kind == "b":   img = ImageEnhance.Brightness(img).enhance(v)
            elif kind == "c": img = ImageEnhance.Contrast(img).enhance(v)
            elif kind == "s": img = ImageEnhance.Color(img).enhance(v)
            elif kind == "h": img = _shift_hue(img, v)
        return img

class ToGrayPIL:
    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.grayscale(img).convert("RGB")

class RandomColorOneOf:
    """Аналог A.OneOf([ColorJitter, ToGray], p=0.3)."""
    def __init__(self, p: float = 0.3, jitter_kwargs=None):
        self.p = p
        self.cj = ColorJitterPIL(**(jitter_kwargs or {}))
        self.gray = ToGrayPIL()
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        return self.cj(img) if random.random() < 0.5 else self.gray(img)


# ─────────────── Аффинные преобразования ───────────────
def _mat_translate(tx: float, ty: float):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=np.float32)

def _mat_scale(sx: float, sy: float):
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0,  0, 1]], dtype=np.float32)

def _mat_shear(shx_deg: float, shy_deg: float):
    shx = math.tan(math.radians(shx_deg))
    shy = math.tan(math.radians(shy_deg))
    return np.array([[1, shx, 0],
                     [shy, 1, 0],
                     [0,  0, 1]], dtype=np.float32)

def _mat_rotate(deg: float):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)

class RandomAffinePIL:
    """
    Аналог A.Affine(scale, translate_percent, rotate, shear, p=...).
    fit_output=False: размер кадра сохраняем.
    """
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.95, 1.05),
        translate_percent: Tuple[Tuple[float, float], Tuple[float, float]] = ((-0.03, 0.03), (-0.03, 0.03)),
        rotate: Tuple[float, float] = (-10.0, 10.0),
        shear: Tuple[float, float] = (-8.0, 8.0),  # применяем по X и по Y независимо
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0),
    ):
        self.p = p
        self.scale_range = scale
        self.tx_range = translate_percent[0]
        self.ty_range = translate_percent[1]
        self.rot_range = rotate
        self.shear_range = shear
        self.resample = resample
        self.fill = fillcolor

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img

        w, h = img.size
        cx, cy = w * 0.5, h * 0.5

        s = random.uniform(*self.scale_range)
        sx = sy = s
        ang = random.uniform(*self.rot_range)
        shx = random.uniform(*self.shear_range)
        shy = random.uniform(*self.shear_range)
        tx = random.uniform(*self.tx_range) * w
        ty = random.uniform(*self.ty_range) * h

        # forward: input → output
        M = np.eye(3, dtype=np.float32)
        M = M @ _mat_translate(-cx, -cy)
        M = M @ _mat_scale(sx, sy)
        M = M @ _mat_shear(shx, shy)
        M = M @ _mat_rotate(ang)
        M = M @ _mat_translate(cx, cy)
        M = M @ _mat_translate(tx, ty)

        # PIL ждёт inverse (output → input)
        Minv = np.linalg.inv(M)
        a, b, c, d, e, f = Minv[0, 0], Minv[0, 1], Minv[0, 2], Minv[1, 0], Minv[1, 1], Minv[1, 2]
        return img.transform(
            (w, h),
            Image.AFFINE,
            data=(a, b, c, d, e, f),
            resample=self.resample,
            fillcolor=self.fill)

class ToTensor:
    def __call__(self, img: Image.Image):
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:  # grayscale → 3ch
            arr = np.stack([arr, arr, arr], axis=-1)
        arr /= 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC→CHW
        return torch.from_numpy(arr)

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)[:, None, None]
        self.std  = torch.tensor(std)[:, None, None]
    def __call__(self, x: torch.Tensor):
        if x.dtype != torch.float32:
            x = x.float()
        return (x - self.mean) / self.std

def build_transforms(img_size: int):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = Compose([
        RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        RandomHorizontalFlip(0.5),
        RandomNoiseOneOf(p=0.15, gaussian_sigma=(5.0, 30.0), sp_amount=(0.01, 0.05)),
        # RandomZeroRect(p=0.1, min_size=4, max_size=32),
        # RandomRotation(15),
         RandomAffinePIL(
            p=0.15,
            scale=(0.95, 1.05),
            translate_percent=((-0.03, 0.03), (-0.03, 0.03)),
            rotate=(-10.0, 10.0),
            shear=(-8.0, 8.0)),
        RandomColorOneOf(p=0.15, jitter_kwargs=dict(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)),
        ToTensor(),
        Normalize(mean, std),
    ])
    val_tf = Compose([
        Resize(int(img_size * 1.14)),
        CenterCrop(img_size),
        ToTensor(),
        Normalize(mean, std),
    ])
    return train_tf, val_tf


# ──────────────── Датасет под root/{split}/{class} ────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def _is_image(p: Path) -> bool: return p.suffix.lower() in IMG_EXTS

class SplitFirstDataset(Dataset):
    """
      root/
        train/{animal,not_animal}/*
        val/{animal,not_animal}/*
    """
    def __init__(self, root: Path, split: str, classes: List[str], pos_class: str, transform=None):
        self.root = root; self.split = split; self.classes = classes
        self.c2i = {c: i for i, c in enumerate(self.classes)}; self.i2c = {v: k for k, v in self.c2i.items()}
        self.pos = pos_class; self.transform = transform
        samples: List[Tuple[Path, int]] = []
        base = root / split
        if not base.exists():
            raise FileNotFoundError(f"Not found: {base}")
        for cls in self.classes:
            d = base / cls
            if not d.exists():
                raise FileNotFoundError(f"Not found: {d}")
            for p in d.rglob("*"):
                if p.is_file() and _is_image(p):
                    samples.append((p, self.c2i[cls]))
        if not samples:
            raise RuntimeError(f"No images found under {base}")
        self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, t = self.samples[i]
        try:
            img = Image.open(p).convert("RGB")
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError):
            return None
        if self.transform is not None:
            img = self.transform(img)
        y = 1 if self.i2c[t] == self.pos else 0
        return img, y


def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)


def build_dataloaders(dataset_root: str, pos: str, img_size: int,
                      batch_size: int, num_workers: int, pin_memory: bool) -> Tuple[DataLoader, DataLoader]:
    root = Path(dataset_root)
    train_tf, val_tf = build_transforms(img_size)
    classes = ["animal", "not_animal"]
    train_ds = SplitFirstDataset(root, "train", classes, pos, transform=train_tf)
    val_ds   = SplitFirstDataset(root, "val",   classes, pos, transform=val_tf)

    common = dict(num_workers=num_workers, pin_memory=True, collate_fn=_collate_skip_none)
    if num_workers > 0:
        common.update(persistent_workers=True, prefetch_factor=4)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True,  **common)
    val_dl   = DataLoader(val_ds,   batch_size, shuffle=False, **common)
    return train_dl, val_dl


# ─────────────── Model / Utils ───────────────
def load_model(module: str) -> nn.Module:
    m = __import__(module)
    cls = getattr(m, "Model", None) or getattr(m, "CNNModel", None)
    if cls is None: raise RuntimeError(f"{module} must export Model or CNNModel")
    return cls()

def logits_1d(out: torch.Tensor) -> torch.Tensor:
    if out.dim() == 2 and out.size(1) == 1: return out[:, 0]
    if out.dim() == 1: return out
    raise RuntimeError(f"Unexpected output shape {tuple(out.shape)}; expected (B,1) or (B,)")

# BCEWithLogits(z,y) == CrossEntropy([0,z], y) → без log_sigmoid → без DML CPU fallback
class CE2FromBinaryLogits(nn.Module):
    def forward(self, z: torch.Tensor, y_float: torch.Tensor) -> torch.Tensor:
        z = z.view(-1)
        two = torch.stack([torch.zeros_like(z), z], dim=1)  # [B,2]
        return F.cross_entropy(two, y_float.long())

@torch.no_grad()
def eval_epoch(model: nn.Module, dl: DataLoader, device: torch.device, crit: nn.Module) -> Tuple[float, float]:
    model.eval()
    tot_loss = tot = correct = 0
    for batch in tqdm(dl, desc="val", leave=False):
        if batch is None: continue
        x, y = batch
        if x.dim() == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)
        lg = logits_1d(model(x))
        loss = crit(lg, y)
        tot_loss += loss.item() * x.size(0)
        pred = (lg >= 0).long()  # эквивалент sigmoid>=0.5
        correct += (pred == y.long()).sum().item()
        tot += x.size(0)
    return tot_loss / max(1, tot), correct / max(1, tot)

def set_seed(seed: Optional[int]):
    if seed is None: return
    import numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

def sd_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


# ─────────────── AdamW без lerp_ (DML-safe) ───────────────
class AdamWNoLerp(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']; beta1, beta2 = group['betas']; eps = group['eps']; wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg = state['exp_avg']; exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                # decoupled weight decay
                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)

                # m_t = beta1*m + (1-beta1)*g   (без foreach/lerp)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta2*v + (1-beta2)*g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-lr)
        return loss


# ──────────────────────── Train ──────────────────────────
def main():
    ap = argparse.ArgumentParser("Train (binary: animal vs not_animal)")
    ap.add_argument("--model", choices=["alexnet", "resnet"], required=True)
    ap.add_argument("--pos", choices=["animal", "not_animal"], required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--outdir", default=r"D:\progs\work\train_model")
    ap.add_argument("--device", choices=["dml", "cuda", "cpu"], default="dml")
    ap.add_argument("--resume-weights", default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--opt", choices=["adamw_nolerp", "sgd"], default="adamw_nolerp")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    tqdm.write(f"[info] device: {device}")

    train_dl, val_dl = build_dataloaders(args.dataset_root, args.pos, args.img_size,
                                         args.batch_size, args.num_workers, pin_memory=True)

    model = load_model(args.model)
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass
    model = model.to(device)

    if args.resume_weights:
        tqdm.write(f"[info] loading {args.resume_weights}")
        inc = model.load_state_dict(torch.load(args.resume_weights, map_location="cpu"), strict=False)
        try:
            tqdm.write(f"[info] missing={list(inc.missing_keys)} unexpected={list(inc.unexpected_keys)}")
        except Exception:
            pass

    crit = CE2FromBinaryLogits()
    if args.opt == "adamw_nolerp":
        opt = AdamWNoLerp(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:  # sgd
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              nesterov=True, weight_decay=args.weight_decay)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = f"{args.model}_{args.pos}_{stamp}"
    csv_path = outdir / f"training_log_{run}.csv"
    weights_path = outdir / f"{run}.pt"

    tr_losses, va_losses, tr_accs, va_accs = [], [], [], []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        f.flush(); os.fsync(f.fileno())

        epbar = tqdm(range(1, args.epochs + 1), desc="epochs", position=0)
        for ep in epbar:
            model.train()
            tot_loss = tot = tot_correct = 0
            bbar = tqdm(train_dl, desc=f"train {ep}/{args.epochs}", position=1, leave=False)
            for batch in bbar:
                if batch is None: continue
                x, y = batch
                if x.dim() == 4:
                    x = x.contiguous(memory_format=torch.channels_last)
                x = x.to(device, non_blocking=True)
                y = y.float().to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                lg = logits_1d(model(x))
                loss = crit(lg, y)
                loss.backward()
                opt.step()

                bs = x.size(0)
                tot_loss += loss.item() * bs
                pred = (lg >= 0).long()
                tot_correct += (pred == y.long()).sum().item()
                tot += bs

                bbar.set_postfix(running_loss=f"{tot_loss/max(1,tot):.4f}",
                                 running_acc=f"{(tot_correct/max(1,tot)):.4f}")
            bbar.close()

            tr_loss = tot_loss / max(1, tot)
            tr_acc  = tot_correct / max(1, tot)
            vl, va  = eval_epoch(model, val_dl, device, crit)

            tr_losses.append(tr_loss); tr_accs.append(tr_acc)
            va_losses.append(vl);      va_accs.append(va)

            wr.writerow([ep, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{vl:.6f}", f"{va:.6f}"])
            f.flush(); os.fsync(f.fileno())
            epbar.set_postfix(train_loss=f"{tr_loss:.4f}", train_acc=f"{tr_acc:.4f}",
                              val_loss=f"{vl:.4f}", val_acc=f"{va:.4f}")
        epbar.close()

    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, weights_path)
    tqdm.write(f"[✓] saved final weights → {weights_path}")

    # Графики
    try:
        import matplotlib.pyplot as plt
        epochs = range(1, len(tr_losses) + 1)
        fig = plt.figure(figsize=(9, 5), dpi=160); ax = fig.add_subplot(111)
        ax.plot(epochs, tr_losses, label="train loss")
        ax.plot(epochs, va_losses, label="val loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(epochs, tr_accs, linestyle=":", label="train acc")
        ax2.plot(epochs, va_accs, linestyle="--", label="val acc")
        ax2.set_ylabel("accuracy")
        l1,lab1=ax.get_legend_handles_labels(); l2,lab2=ax2.get_legend_handles_labels()
        fig.legend(l1+l2, lab1+lab2, loc="lower left")
        fig.suptitle(f"{args.model.upper()} | pos={args.pos} | {stamp}")
        png_path = outdir / f"curves_{run}.png"
        fig.tight_layout(); fig.savefig(png_path); plt.close(fig)
        tqdm.write(f"[✓] curves saved → {png_path}")
    except Exception as e:
        tqdm.write(f"[warn] could not save curves: {e}")

    tqdm.write(f"[✓] csv saved → {csv_path}")
    tqdm.write(f"[✓] artifacts dir → {outdir}")


if __name__ == "__main__":
    main()
