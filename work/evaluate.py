import argparse, csv, math, random, importlib
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch, matplotlib.pyplot as plt

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def stamp(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def sanitize(s): 
    for ch in '<>:"/\\|?*': s=s.replace(ch,"_")
    return s
def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    try: import torch_directml; return torch_directml.device()
    except: return torch.device("cpu")

def load_model(mod_name, weights, dev):
    m = importlib.import_module(mod_name)
    if not hasattr(m,"CNNModel"): raise RuntimeError(f"CNNModel not found in {mod_name}")
    model = m.CNNModel()
    state = torch.load(str(weights), map_location="cpu")
    if isinstance(state,dict):
        for k in ("model_state_dict","state_dict"):
            if k in state and isinstance(state[k],dict): state = state[k]; break
    model.load_state_dict(state, strict=False)
    return model.to(dev).eval()

def pil_to_tensor(p, size):
    x = Image.open(p).convert("RGB").resize((size,size), Image.BILINEAR)
    a = np.asarray(x, dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0)

def list_images(root: Path):
    if not root.exists(): return []
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

def find_split(root: Path, pos: str, split: str):
    neg = f"not_{pos}"
    cands = [root/split/pos, root/split/neg]
    if not any(c.exists() for c in cands): cands = [root/pos/split, root/neg/split]
    pos_dir, neg_dir = cands[0].parent/pos, cands[1].parent/neg
    return [(p,1) for p in list_images(pos_dir)], [(p,0) for p in list_images(neg_dir)]

def auc_by_ranks(y_true, y_prob):
    y_true = y_true.astype(int)
    n1, n0 = int((y_true==1).sum()), int((y_true==0).sum())
    if n1==0 or n0==0: return float("nan")
    order = np.argsort(y_prob, kind="mergesort")
    ranks = np.empty_like(order, dtype=float); i=0
    while i<len(order):
        j=i
        while j+1<len(order) and y_prob[order[j+1]]==y_prob[order[i]]: j+=1
        r=0.5*(i+j)+1.0
        for k in range(i,j+1): ranks[order[k]]=r
        i=j+1
    return float((ranks[y_true==1].sum() - n1*(n1+1)/2)/(n1*n0))

def basic_metrics(y_true, y_prob, thr):
    y_pred = (y_prob>=thr).astype(int)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    prec = tp/(tp+fp) if tp+fp>0 else 0.0
    rec  = tp/(tp+fn) if tp+fn>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    acc  = (tp+tn)/max(1,tp+tn+fp+fn)
    return {"tp":tp,"tn":tn,"fp":fp,"fn":fn,"precision":prec,"recall":rec,"f1":f1,"accuracy":acc,"roc_auc":auc_by_ranks(y_true,y_prob)}

def roc_curve_pts(y_true: np.ndarray, y_prob: np.ndarray):
    thr = np.r_[np.inf, np.unique(y_prob)[::-1], -np.inf]
    P = (y_true==1).sum(); N = (y_true==0).sum()
    tpr, fpr = [], []
    for t in thr:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_true==1)&(y_pred==1)).sum()
        fp = ((y_true==0)&(y_pred==1)).sum()
        fn = P - tp; tn = N - fp
        tpr.append( tp / P if P>0 else 0.0 )
        fpr.append( fp / N if N>0 else 0.0 )
    return np.array(fpr), np.array(tpr)


def save_metrics_png(outdir, model_name, weights_path, pos, dt, m, thr, y_true, y_prob):
    prefix = f"{sanitize(model_name)}_{sanitize(weights_path.stem)}_{pos}"
    p = Path(outdir)/f"{prefix}_метрики.png"

    # Confusion matrices
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_true==1)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())

    cm_abs = np.array([[tn, fp],[fn, tp]], dtype=float)
    row_sums = cm_abs.sum(axis=1, keepdims=True); row_sums[row_sums==0]=1.0
    cm_norm = cm_abs / row_sums

    # ROC
    fpr, tpr = roc_curve_pts(y_true, y_prob)

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(10,8), dpi=140, constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1,1], width_ratios=[1,1])
    ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0]); ax4 = fig.add_subplot(gs[1,1])

    # CM abs
    im1 = ax1.imshow(cm_abs, cmap="viridis")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{int(cm_abs[i,j])}", ha="center", va="center", color="w", fontsize=10)
    ax1.set_xticks([0,1], ["neg","pos"]); ax1.set_yticks([0,1], ["neg","pos"])
    ax1.set_title("Confusion Matrix (abs)"); ax1.set_aspect("equal")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # CM norm
    im2 = ax2.imshow(cm_norm, vmin=0, vmax=1, cmap="viridis")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center", color="w", fontsize=10)
    ax2.set_xticks([0,1], ["neg","pos"]); ax2.set_yticks([0,1], ["neg","pos"])
    ax2.set_title("Confusion Matrix (normalized)"); ax2.set_aspect("equal")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # ROC
    ax3.plot(fpr, tpr, lw=2)
    ax3.plot([0,1],[0,1],"--", lw=1)
    ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR"); ax3.set_title("ROC curve")
    ax3.grid(alpha=0.3)

    # Text block
    ax4.axis("off")
    text = (
        f"threshold: {thr:.3f}\n"
        f"accuracy : {m['accuracy']:.4f}\n"
        f"precision: {m['precision']:.4f}\n"
        f"recall   : {m['recall']:.4f}\n"
        f"F1-score : {m['f1']:.4f}\n"
        f"ROC AUC  : {'nan' if math.isnan(m['roc_auc']) else f'{m['roc_auc']:.4f}'}\n\n"
        f"TP={tp}  FP={fp}\nFN={fn}  TN={tn}"
    )
    ax4.text(0.02, 0.98, text, va="top", ha="left", fontsize=11)

    title = f"Модель: {model_name} | Веса: {weights_path.name} | pos: {pos} | дата: {dt}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(p); plt.close(fig); return p


def save_check_tile_png(outdir, model_name, weights_path, pos, dt, model, dev, check_dir, size, thr):
    prefix = f"{sanitize(model_name)}_{sanitize(weights_path.stem)}_{pos}"
    p = Path(outdir)/f"{prefix}_проверка.png"
    files = list_images(Path(check_dir))
    fig, axes = plt.subplots(4,4, figsize=(10,10), dpi=140, constrained_layout=False); axes = axes.flatten()
    if not files:
        for ax in axes: ax.axis("off")
        axes[7].text(0.5,0.5,f"Нет изображений в {check_dir}",ha="center",va="center",fontsize=14)
    else:
        random.shuffle(files); files = files[:16]
        with torch.no_grad():
            for ax, f in zip(axes, files):
                try: t = pil_to_tensor(f, size).to(dev)
                except (OSError, UnidentifiedImageError): ax.axis("off"); continue
                prob = torch.sigmoid(model(t)).item()
                ax.imshow(Image.open(f).convert("RGB").resize((size,size), Image.BILINEAR))
                ax.set_title(f"{'pos' if prob>=thr else 'not_pos'} (p={prob:.2f})", fontsize=9); ax.axis("off")
            for k in range(len(files), len(axes)): axes[k].axis("off")
    title = f"Модель: {model_name} | Веса: {weights_path.name} | pos: {pos} | дата: {dt}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(p); plt.close(fig); return p

def save_csv(outdir, prefix, model_name, pos, dt, thr, m):
    p = Path(outdir)/f"{prefix}_metrics.csv"
    with open(p,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","pos","date","threshold","tp","fp","tn","fn","precision","recall","f1","accuracy","roc_auc"])
        w.writerow([model_name,pos,dt,f"{thr:.4f}",m["tp"],m["fp"],m["tn"],m["fn"],
                    f"{m['precision']:.6f}",f"{m['recall']:.6f}",f"{m['f1']:.6f}",
                    f"{m['accuracy']:.6f}", "nan" if math.isnan(m["roc_auc"]) else f"{m['roc_auc']:.6f}"])
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pos", default="cat")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--check_dir", default=r"D:\progs\work\datasets\dataset_for_cat\check_dataset")
    ap.add_argument("--outdir", default="eval_out")
    a = ap.parse_args()

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    dev = device()
    model = load_model(a.model, Path(a.weights), dev)
    pos_list, neg_list = find_split(Path(a.data), a.pos.lower(), a.split)
    pairs = pos_list + neg_list
    if not pairs: raise RuntimeError(f"No images for split='{a.split}' in {a.data}")

    y_true, y_prob = [], []
    with torch.no_grad():
        for p, lab in pairs:
            try: t = pil_to_tensor(p, a.img_size).to(dev)
            except (OSError, UnidentifiedImageError): continue
            y_true.append(lab); y_prob.append(torch.sigmoid(model(t)).item())
    y_true, y_prob = np.array(y_true,int), np.array(y_prob,float)

    m = basic_metrics(y_true, y_prob, a.threshold)
    prefix = f"{sanitize(a.model)}_{sanitize(Path(a.weights).stem)}_{a.pos}"
    dt = stamp()

    csv_p = save_csv(outdir, prefix, a.model, a.pos, dt, a.threshold, m)
    met_p = save_metrics_png(outdir, a.model, Path(a.weights), a.pos, dt, m, a.threshold, y_true, y_prob)
    tile_p = save_check_tile_png(outdir, a.model, Path(a.weights), a.pos, dt, model, dev, a.check_dir, a.img_size, a.threshold)

    print(f"[saved] {met_p}"); print(f"[saved] {tile_p}"); print(f"[saved] {csv_p}")

if __name__ == "__main__": main()
