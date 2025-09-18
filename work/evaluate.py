# evaluate.py
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ───────────────────────── device ─────────────────────────
def get_device(name: str) -> torch.device:
    name = name.lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "dml":
        try:
            import torch_directml  # noqa: F401
            return torch_directml.device()
        except Exception as e:
            print(f"[warn] torch-directml not available ({e}); fallback to CPU.")
    return torch.device("cpu")


# ──────────────────────── transforms ───────────────────────
def _default_val_transform(img_size: int):
    from torchvision import transforms as T
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std),
    ])


def build_val_transform(img_size: int):
    try:
        # Try external transform.py (preferred)
        from transform import build_transforms, AugConfig  # type: ignore
        train_tf, val_tf = build_transforms(AugConfig(img_size=img_size))
        return val_tf
    except Exception:
        return _default_val_transform(img_size)


# ─────────────────────── dataset / loader ───────────────────────
def _safe_imagefolder_val(root: Path, pos: str, transform) -> "torch.utils.data.Dataset":
    from torchvision.datasets import ImageFolder
    from PIL import UnidentifiedImageError, Image

    class SafeImageFolder(ImageFolder):
        def __init__(self, root, **kwargs):
            super().__init__(root, **kwargs)
            self._pos_name = pos
            self._class_to_idx = self.class_to_idx
            self._idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        def __getitem__(self, idx):
            path, target = self.samples[idx]
            try:
                img = self.loader(path)  # PIL
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                return None
            if self.transform is not None:
                img = self.transform(img)
            cls_name = self._idx_to_class[target]
            y = 1 if cls_name == self._pos_name else 0
            return img, y, path

    return SafeImageFolder(root, transform=transform)


def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys, ps = zip(*batch)
    return torch.utils.data.dataloader.default_collate(xs), torch.tensor(ys), list(ps)


def resolve_split_dir(dataset_root: str | Path, split: str = "val") -> Path:
    """
    Возвращает каталог сплита для ImageFolder.
    Работает и когда в --dataset-root уже передан .../val, и когда передан корень задачи (.../cat).
    Также понимает валидные синонимы 'valid'/'validation'.
    """
    root = Path(dataset_root)
    split_norm = split.lower()
    split_aliases = {"val", "valid", "validation", "train", "test"}

    # 1) Если нам уже дали путь на сплит — используем как есть.
    if root.name.lower() in split_aliases and root.is_dir():
        return root

    # 2) Стандартный случай: внутри лежит нужный сплит.
    cand = root / split_norm
    if cand.is_dir():
        return cand

    # 3) Синонимы 'val'
    for alias in ("valid", "validation"):
        cand = root / alias
        if cand.is_dir():
            return cand

    # 4) Если в корне напрямую лежат классовые папки — считаем этот уровень сплитом.
    class_dirs = {"cat", "not_cat", "dog", "not_dog"}
    if any((root / d).is_dir() for d in class_dirs):
        return root

    # 5) Не нашли — понятная ошибка.
    raise FileNotFoundError(
        f"Не найден каталог сплита в '{root}'. Ожидалось: '{root}/val' (или valid/validation) "
        f"или сами классы на этом уровне."
    )


def build_val_loader(dataset_root: str, pos: str, img_size: int,
                     batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    val_dir = resolve_split_dir(dataset_root, split="val")
    print(f"[info] val dir: {val_dir}")
    tf = build_val_transform(img_size)
    ds = _safe_imagefolder_val(val_dir, pos=pos, transform=tf)

    # лог классов
    if hasattr(ds, "classes"):
        print(f"[info] classes: {ds.classes}  |  samples: {len(ds.samples)}")

    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=_collate_skip_none
    )


# ─────────────────────── model loader ───────────────────────
def load_model(module_name: str) -> nn.Module:
    import importlib
    mod = importlib.import_module(module_name)
    model_cls = getattr(mod, "CNNModel", None) or getattr(mod, "Model", None)
    if model_cls is None:
        raise RuntimeError(f"Module {module_name} must export CNNModel or Model class.")
    return model_cls()


# ─────────────────────── metrics helpers ───────────────────────
def confusion_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Binary ROC; ties handled by stable sort
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    tpr = tps / (P if P else 1)
    fpr = fps / (N if N else 1)

    # prepend (0,0), append (1,1)
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return fpr, tpr


def auc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


# ───────────────────────── evaluation ─────────────────────────
@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device: torch.device
                  ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    probs: List[float] = []
    trues: List[int] = []
    paths: List[str] = []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        if batch is None:
            continue
        x, y, p = batch
        x = x.to(device)
        logits = model(x)
        # поддержка форм (B,1) и (B,)
        if logits.ndim == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(prob.tolist())
        trues.extend(y.numpy().tolist())
        paths.extend(p)
    return np.array(trues, dtype=np.int64), np.array(probs, dtype=np.float32), paths


def save_metrics_artifacts(
    outdir: Path,
    prefix: str,
    model_name: str,
    pos: str,
    stamp: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> None:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp, fp, tn, fn = confusion_from_preds(y_true, y_pred)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)

    fpr, tpr = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # CSV (summary)
    csv_summary = outdir / f"{prefix}_metrics.csv"
    with open(csv_summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pos", "date", "threshold", "tp", "fp", "tn", "fn",
                    "precision", "recall", "f1", "accuracy", "roc_auc"])
        w.writerow([model_name, pos, stamp, f"{threshold:.4f}", tp, fp, tn, fn,
                    f"{precision:.6f}", f"{recall:.6f}", f"{f1:.6f}",
                    f"{accuracy:.6f}", f"{roc_auc:.6f}"])

    # JSON (summary)
    json_summary = outdir / f"{prefix}_metrics.json"
    with open(json_summary, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "pos": pos,
            "date": stamp,
            "threshold": threshold,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "roc_auc": roc_auc
        }, f, indent=2, ensure_ascii=False)

    # PNG tile
    try:
        import matplotlib.pyplot as plt

        cm = np.array([[tn, fp],
                       [fn, tp]], dtype=np.int64)
        # normalize by row
        cm_norm = cm.astype(np.float64)
        for r in range(2):
            s = cm_norm[r].sum()
            cm_norm[r] = cm_norm[r] / s if s else cm_norm[r]

        fig = plt.figure(figsize=(10, 8), dpi=160)

        # CM absolute
        ax1 = fig.add_subplot(2, 2, 1)
        im1 = ax1.imshow(cm, interpolation="nearest")
        ax1.set_title("Confusion Matrix (abs)")
        ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
        ax1.set_xticklabels(["neg", "pos"]); ax1.set_yticklabels(["neg", "pos"])
        for (i, j), v in np.ndenumerate(cm):
            ax1.text(j, i, str(v), ha="center", va="center")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # CM normalized
        ax2 = fig.add_subplot(2, 2, 2)
        im2 = ax2.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=1)
        ax2.set_title("Confusion Matrix (normalized)")
        ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
        ax2.set_xticklabels(["neg", "pos"]); ax2.set_yticklabels(["neg", "pos"])
        for (i, j), v in np.ndenumerate(cm_norm):
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # ROC curve
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
        ax3.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax3.set_xlabel("FPR")
        ax3.set_ylabel("TPR")
        ax3.set_title("ROC curve")
        ax3.legend(loc="lower right")
        ax3.grid(True, alpha=0.3)

        # Metrics textbox
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        lines = [
            f"threshold: {threshold:.3f}",
            f"accuracy : {accuracy:.4f}",
            f"precision: {precision:.4f}",
            f"recall   : {recall:.4f}",
            f"F1-score : {f1:.4f}",
            f"ROC AUC  : {roc_auc:.4f}",
            "",
            f"TP={tp}  FP={fp}",
            f"FN={fn}  TN={tn}",
        ]
        ax4.text(0.0, 0.9, "\n".join(lines), va="top", ha="left", fontsize=11)

        fig.suptitle(f"{model_name.upper()} | pos={pos} | {stamp}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])

        png_path = outdir / f"{prefix}_metrics.png"
        fig.savefig(png_path)
        plt.close(fig)
        print(f"[✓] saved metrics tile → {png_path}")
    except Exception as e:
        print(f"[warn] could not render PNG metrics tile: {e}")


def save_predictions_csv(outdir: Path, prefix: str, paths: List[str],
                         y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:
    y_pred = (y_prob >= threshold).astype(np.int64)
    csv_path = outdir / f"{prefix}_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "y_true", "y_prob", "y_pred"])
        for p, yt, yp, yhat in zip(paths, y_true.tolist(), y_prob.tolist(), y_pred.tolist()):
            w.writerow([p, yt, f"{yp:.6f}", yhat])


# ─────────────────────────── CLI ───────────────────────────
def main():
    ap = argparse.ArgumentParser("Evaluate binary classifier (pos vs background)")
    ap.add_argument("--model", choices=["alexnet", "resnet"], required=True)
    ap.add_argument("--pos", choices=["cat", "dog"], required=True)
    ap.add_argument("--dataset-root", required=True, help=r'Root like "D:\progs\work\datasets\dataset_for_cat\cat" or ".../cat/val"')
    ap.add_argument("--weights", required=True, help="Path to saved .pt state_dict.")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", choices=["dml", "cuda", "cpu"], default="dml")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--outdir", default=r"D:\progs\work\train_model")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"[info] device: {device}")

    # loader
    val_loader = build_val_loader(
        dataset_root=args.dataset_root,
        pos=args.pos,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # model
    model = load_model(args.model)

    # загрузка весов (robust)
    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "weights", "net"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    # снять префикс 'module.' если был DDP/EMA
    if isinstance(state, dict):
        state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    model.load_state_dict(state, strict=False)
    model.to(device)

    # inference
    y_true, y_prob, paths = run_inference(model, val_loader, device)

    # outputs
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.prefix or f"{args.model}_{args.pos}_{stamp}"

    save_predictions_csv(outdir, prefix, paths, y_true, y_prob, args.threshold)
    save_metrics_artifacts(
        outdir=outdir, prefix=prefix,
        model_name=args.model, pos=args.pos, stamp=stamp,
        y_true=y_true, y_prob=y_prob, threshold=args.threshold
    )

    print("[✓] evaluation complete")


if __name__ == "__main__":
    main()
