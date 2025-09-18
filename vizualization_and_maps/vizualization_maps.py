# evaluate.py — бинарная оценка: pos (1) против non-pos (0)
import argparse, csv, json, importlib
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ───────────────── device ─────────────────
def get_device(name: str) -> torch.device:
    name = name.lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "dml":
        try:
            import torch_directml  # noqa
            return torch_directml.device()
        except Exception as e:
            print(f"[warn] torch-directml not available ({e}); fallback to CPU.")
    return torch.device("cpu")

# ─────────────── transforms ───────────────
def _default_val_transform(img_size: int):
    from torchvision import transforms as T
    mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

def build_val_transform(img_size: int):
    try:
        from transform import build_transforms, AugConfig  # type: ignore
        _train, val_tf = build_transforms(AugConfig(img_size=img_size))
        return val_tf
    except Exception:
        return _default_val_transform(img_size)

# ───────────── dataset helpers ────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def _gather_images(root: Path) -> List[Path]:
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def _resolve_split_dir(dataset_root: str | Path, split: str = "val") -> Path:
    """
    Возвращает каталог сплита.
      • .../<task_root>/<split>
      • .../<task_root>  (если на этом уровне уже лежат class-сабпапки)
      • .../<pos>/<split>  (если дан путь на класс)
    """
    root = Path(dataset_root)
    # указан уже сплит
    if root.name.lower() in {"val", "valid", "validation", "train", "test"}:
        return root
    # стандартно: есть подпапка сплита
    if (root / split).is_dir():
        return root / split
    # на уровне root уже лежат {class} подпапки
    if any((root / d).is_dir() for d in ("cat", "not_cat", "dog", "not_dog")):
        return root
    raise FileNotFoundError(
        f"Не найден каталог сплита в '{root}'. Ожидалось '{root}/{split}' или наличие подпапок классов."
    )

def _auto_find_nonpos_dir(pos_dir: Path, split_dir: Path, explicit_nonpos: str | None) -> Path:
    """
    Возвращает директорию non-pos, стараясь угадать имя:
    not_<pos>, non_<pos>, other, others, bg, background, negative.
    Если явно задано --nonpos-name, используем его.
    """
    candidates = []
    if explicit_nonpos:
        candidates.append(explicit_nonpos)

    pos_name = pos_dir.name.lower()
    parent = pos_dir.parent

    guesses = [
        f"not_{pos_name}",
        f"non_{pos_name}",
        "other", "others", "bg", "background", "negative", "nonpos", "non-pos", "notpos"
    ]
    candidates.extend(guesses)

    # 1) если split_dir = .../<pos>/<split> → пробуем .../<nonpos>/<split>
    for cand in candidates:
        cand_path = parent / cand / split_dir.name
        if cand_path.is_dir():
            return cand_path

    # 2) иначе пробуем на уровне соседей без split-а
    for cand in candidates:
        cand_path = parent / cand
        if cand_path.is_dir():
            # если внутри есть подпапка split — используем её
            if (cand_path / split_dir.name).is_dir():
                return cand_path / split_dir.name
            return cand_path

    # 3) последний шанс: если в parent ровно две директории, берём «не pos»
    siblings = [d for d in parent.iterdir() if d.is_dir()]
    others = [d for d in siblings if d.name.lower() != pos_name]
    if len(others) == 1:
        # учитываем возможный split
        if (others[0] / split_dir.name).is_dir():
            return (others[0] / split_dir.name)
        return others[0]

    raise FileNotFoundError(f"Не удалось автоматически найти non-pos рядом с '{pos_dir}'.")

# ─────────────── датасеты ────────────────
class _FlatBinaryValDataset(Dataset):
    """
    Плоская разметка: две папки с изображениями pos и non-pos (на одном уровне).
    """
    def __init__(self, pos_dir: Path, nonpos_dir: Path, transform):
        from PIL import Image
        self.Image = Image
        self.transform = transform
        self.pos_paths = _gather_images(pos_dir)
        self.neg_paths = _gather_images(nonpos_dir)
        self.samples: List[tuple[Path, int]] = [(p, 1) for p in self.pos_paths] + [(n, 0) for n in self.neg_paths]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import UnidentifiedImageError
        path, y = self.samples[idx]
        try:
            img = self.Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, self.Image.DecompressionBombError):
            return None
        if self.transform is not None:
            img = self.transform(img)
        return img, y, str(path)

def _imagefolder_binary(root: Path, pos_name: str, transform) -> Dataset:
    """
    Классический ImageFolder с переоценкой меток в {pos:1, non-pos:0}.
    """
    from torchvision.datasets import ImageFolder
    from PIL import UnidentifiedImageError, Image

    class SafeIF(ImageFolder):
        def __init__(self, root, **kwargs):
            super().__init__(root, **kwargs)
            self._pos_name = pos_name
            self._idx2name = {v: k for k, v in self.class_to_idx.items()}

        def __getitem__(self, index):
            path, target = self.samples[index]
            try:
                img = self.loader(path)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                return None
            if self.transform is not None:
                img = self.transform(img)
            y = 1 if self._idx2name[target] == self._pos_name else 0
            return img, y, path

    return SafeIF(root, transform=transform)

def build_val_loader(dataset_root: str, pos: str, img_size: int,
                     batch_size: int, num_workers: int, pin_memory: bool,
                     nonpos_name: str | None) -> DataLoader:
    split_dir = _resolve_split_dir(dataset_root, split="val")
    tf = build_val_transform(img_size)
    print(f"[info] split dir: {split_dir}")

    # Схема А: split_dir содержит подпапки классов (ImageFolder)
    subdirs = [d.name for d in split_dir.iterdir() if d.is_dir()]
    if subdirs:
        print(f"[info] layout=ImageFolder; subdirs={subdirs}")
        ds = _imagefolder_binary(split_dir, pos_name=pos, transform=tf)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory,
                          collate_fn=_collate_skip_none)

    # Схема Б: pos/val/* и рядом non-pos/val/*
    # Если нам дали путь на .../<task_root>/<pos> ИЛИ .../<task_root>/<pos>/<split>
    # — достроим non-pos соседей.
    # Найдём директорию pos (если split_dir — это <pos>/<split>, то pos_dir = parent)
    if split_dir.name.lower() in {"val", "valid", "validation"}:
        pos_dir = split_dir.parent
    else:
        pos_dir = split_dir  # уже на уровне класса

    nonpos_dir = _auto_find_nonpos_dir(pos_dir=pos_dir, split_dir=split_dir, explicit_nonpos=nonpos_name)
    # привести к реальным каталогам со сплитом
    pos_imgs_dir = split_dir if split_dir.name.lower() in {"val", "valid", "validation"} else split_dir
    nonpos_imgs_dir = nonpos_dir
    print(f"[info] layout=flat; pos={pos_imgs_dir} | non-pos={nonpos_imgs_dir}")

    ds = _FlatBinaryValDataset(pos_imgs_dir, nonpos_imgs_dir, tf)
    print(f"[info] samples: pos={len(ds.pos_paths)} non-pos={len(ds.neg_paths)} total={len(ds)}")
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory,
                      collate_fn=_collate_skip_none)

def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys, ps = zip(*batch)
    return torch.utils.data.dataloader.default_collate(xs), torch.tensor(ys), list(ps)

# ─────────────── model loader ─────────────
def load_model(module_name: str) -> nn.Module:
    mod = importlib.import_module(module_name)
    model_cls = getattr(mod, "CNNModel", None) or getattr(mod, "Model", None)
    if model_cls is None:
        raise RuntimeError(f"Module {module_name} must export CNNModel or Model class.")
    return model_cls()

# ───────────── metrics helpers ────────────
def confusion_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]; y_score = y_score[order]
    P = np.sum(y_true == 1); N = np.sum(y_true == 0)
    tps = np.cumsum(y_true == 1); fps = np.cumsum(y_true == 0)
    tpr = tps / (P if P else 1); fpr = fps / (N if N else 1)
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return fpr, tpr

def auc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))

# ─────────────── evaluation ───────────────
@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader, device: torch.device
                  ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    probs, trues, paths = [], [], []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        if batch is None:
            continue
        x, y, p = batch
        x = x.to(device)
        logits = model(x)
        if logits.ndim == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(prob.tolist()); trues.extend(y.numpy().tolist()); paths.extend(p)
    return np.array(trues, np.int64), np.array(probs, np.float32), paths

def save_metrics_artifacts(out: Path, prefix: str, model_name: str, pos: str, stamp: str,
                           y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> None:
    y_pred = (y_prob >= thr).astype(np.int64)
    tp, fp, tn, fn = confusion_from_preds(y_true, y_pred)
    precision = safe_div(tp, tp + fp); recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    fpr, tpr = roc_curve(y_true, y_prob); roc_auc = auc(fpr, tpr)

    # CSV
    with open(out / f"{prefix}_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","pos","date","threshold","tp","fp","tn","fn",
                    "precision","recall","f1","accuracy","roc_auc"])
        w.writerow([model_name, pos, stamp, f"{thr:.4f}", tp, fp, tn, fn,
                    f"{precision:.6f}", f"{recall:.6f}", f"{f1:.6f}",
                    f"{accuracy:.6f}", f"{roc_auc:.6f}"])

    # JSON
    with open(out / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name, "pos": pos, "date": stamp, "threshold": thr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "roc_auc": roc_auc
        }, f, indent=2, ensure_ascii=False)

    # PNG (CM + ROC)
    try:
        import matplotlib.pyplot as plt
        cm = np.array([[tn, fp],[fn, tp]], dtype=np.int64)
        cmn = cm.astype(np.float64); cmn[0] /= (cm[0].sum() or 1); cmn[1] /= (cm[1].sum() or 1)

        fig = plt.figure(figsize=(10,8), dpi=160)
        ax1 = fig.add_subplot(2,2,1)
        im1 = ax1.imshow(cm, interpolation="nearest")
        ax1.set_title("Confusion Matrix (abs)")
        ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
        ax1.set_xticklabels(["non-pos","pos"]); ax1.set_yticklabels(["non-pos","pos"])
        for (i,j),v in np.ndenumerate(cm): ax1.text(j,i,str(v),ha="center",va="center")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(2,2,2)
        im2 = ax2.imshow(cmn, interpolation="nearest", vmin=0, vmax=1)
        ax2.set_title("Confusion Matrix (normalized)")
        ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
        ax2.set_xticklabels(["non-pos","pos"]); ax2.set_yticklabels(["non-pos","pos"])
        for (i,j),v in np.ndenumerate(cmn): ax2.text(j,i,f"{v:.2f}",ha="center",va="center")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(2,2,3)
        ax3.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax3.plot([0,1],[0,1],"--",linewidth=1)
        ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR"); ax3.set_title("ROC")
        ax3.legend(loc="lower right"); ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(2,2,4); ax4.axis("off")
        txt = "\n".join([
            f"threshold: {thr:.3f}",
            f"accuracy : {accuracy:.4f}",
            f"precision: {precision:.4f}",
            f"recall   : {recall:.4f}",
            f"F1-score : {f1:.4f}",
            f"ROC AUC  : {roc_auc:.4f}",
            "",
            f"TP={tp}  FP={fp}",
            f"FN={fn}  TN={tn}",
        ])
        ax4.text(0.0,0.9,txt,va="top",ha="left",fontsize=11)

        fig.suptitle(f"{model_name.upper()} | pos={pos} | {stamp}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0,0.02,1,0.95])
        fig.savefig(out / f"{prefix}_metrics.png"); plt.close(fig)
    except Exception as e:
        print(f"[warn] could not render PNG metrics tile: {e}")

def save_predictions_csv(out: Path, prefix: str, paths: List[str],
                         y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> None:
    y_pred = (y_prob >= thr).astype(np.int64)
    with open(out / f"{prefix}_predictions.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path","y_true","y_prob","y_pred"])
        for p, yt, yp, yhat in zip(paths, y_true.tolist(), y_prob.tolist(), y_pred.tolist()):
            w.writerow([p, yt, f"{yp:.6f}", yhat])

# ──────────────────── CLI ────────────────────
def main():
    ap = argparse.ArgumentParser("Evaluate: pos vs non-pos")
    ap.add_argument("--model", choices=["alexnet","resnet"], required=True)
    ap.add_argument("--pos", required=True, help="Имя положительного класса (напр., cat или dog)")
    ap.add_argument("--nonpos-name", default=None,
                    help="Необязательное имя папки для non-pos; иначе определяется автоматически")
    ap.add_argument("--dataset-root", required=True,
                    help=r'Корень задачи: ".../dataset_for_cat/cat" ИЛИ путь прямо на сплит ".../cat/val"')
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", choices=["dml","cuda","cpu"], default="dml")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--outdir", default=r"D:\progs\work\train_model")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"[info] device: {device}")

    # data loader
    val_loader = build_val_loader(
        dataset_root=args.dataset_root,
        pos=args.pos,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        nonpos_name=args.nonpos_name,
    )

    # model
    model = load_model(args.model)
    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict):
        for k in ("state_dict","model","weights","net"):
            if k in state and isinstance(state[k], dict):
                state = state[k]; break
    if isinstance(state, dict):
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)

    # inference
    y_true, y_prob, paths = run_inference(model, val_loader, device)

    # save
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.prefix or f"{args.model}_{args.pos}_{stamp}"

    save_predictions_csv(outdir, prefix, paths, y_true, y_prob, args.threshold)
    save_metrics_artifacts(outdir, prefix, args.model, args.pos, stamp, y_true, y_prob, args.threshold)
    print("[✓] evaluation complete")

if __name__ == "__main__":
    main()
