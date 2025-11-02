# train_stages.py
import argparse, os, sys, shutil, json, importlib, inspect
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError
import torch, torch.nn as nn, torch.nn.functional as F

# --------------------------- small utils ---------------------------
IMG_EXTS={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff",".gif"}
def imgs(d): return sorted([p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXTS])
def clean(path): p=Path(path); 
# recreate dir
def clean(path):
    p=Path(path)
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def get_device(name:str):
    name=(name or "cpu").lower()
    if name=="cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if name=="dml":
        try: import torch_directml; return torch_directml.device()
        except: pass
    return torch.device("cpu")

def load_model(modname:str)->nn.Module:
    m=importlib.import_module(modname)
    cls=(getattr(m,"CNNModel",None) or getattr(m,"Model",None) or
         next(c for _,c in inspect.getmembers(m,inspect.isclass)
              if issubclass(c,nn.Module) and c.__module__==m.__name__))
    return cls()

def module_by_path(root:nn.Module, path:str)->nn.Module:
    o=root
    for t in path.split("."):
        if "[" in t and t.endswith("]"):
            b,i=t[:-1].split("["); o=getattr(o,b)[int(i)]
        else: o=getattr(o,t)
    if not isinstance(o,nn.Module): raise ValueError(path)
    return o

# геометрия как в val (Resize(1.14*S) → CenterCrop(S))
def resize_shorter(pil, shorter):
    w,h=pil.size
    if w<h: nw,nh=shorter, int(round(h*shorter/w))
    else:   nh,nw=shorter, int(round(w*shorter/h))
    return pil.resize((nw,nh), Image.Resampling.BICUBIC)
def center_crop(pil, size):
    w,h=pil.size; i=max(0,(h-size)//2); j=max(0,(w-size)//2)
    return pil.crop((j,i,j+size,i+size))
def align_like_val(pil, size, k=1.14): return center_crop(resize_shorter(pil, int(round(size*k))), size)

def find_latest(outdir, model, pos):
    ps=sorted(Path(outdir).glob(f"{model}_{pos}_*.pt"), key=lambda p:p.stat().st_mtime, reverse=True)
    return ps[0] if ps else None

# --------------------------- heatmaps ---------------------------
@torch.no_grad()
def make_heatmaps(model_name, weights, layer, src_dir, out_dir, img_size, device):
    try:
        from transform import build_val_transform as _bv
        tf=_bv(img_size)
    except Exception:
        from torchvision import transforms as T
        tf=T.Compose([T.Resize(int(img_size*1.14)), T.CenterCrop(img_size), T.ToTensor(),
                      T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    clean(out_dir)
    net=load_model(model_name).to(device).eval()
    # безопасная загрузка весов
    try: st=torch.load(str(weights), map_location="cpu", weights_only=True)
    except TypeError: st=torch.load(str(weights), map_location="cpu")
    net.load_state_dict(st["model"] if isinstance(st,dict) and "model" in st else st, strict=False)

    probe=module_by_path(net, layer); fmap_cpu=None
    def hk(_m,_i,o): nonlocal fmap_cpu; fmap_cpu=o.detach().to("cpu")
    h=probe.register_forward_hook(hk)

    for p in imgs(src_dir):
        try: im=Image.open(p).convert("RGB")
        except (UnidentifiedImageError,OSError): continue
        _=net(tf(im).unsqueeze(0).to(device))
        if fmap_cpu is None: continue
        s=F.relu(fmap_cpu).sum(1,keepdim=True)
        up=F.interpolate(s, size=(img_size,img_size), mode="bilinear", align_corners=False)[0,0]
        a=(up-up.min())/(up.max()-up.min()+1e-12)
        stem=p.stem; arr=a.numpy().astype(np.float32)
        np.save(Path(out_dir,f"{stem}.heat.npy"), arr)
        Image.fromarray((arr*255).astype(np.uint8),"L").save(Path(out_dir,f"{stem}.heat.gray.png"))
    h.remove()
    Path(out_dir,"_meta.json").write_text(json.dumps({"model":model_name,"weights":str(weights),"layer":layer},ensure_ascii=False),encoding="utf-8")

# --------------------------- masks / erase ---------------------------
def mask_from_heat(a01:np.ndarray, mode:str, percent:float, alpha:float, dilate:int):
    if mode=="alpha": m=a01 >= (float(a01.max())*max(0.0,min(1.0,alpha)))
    elif mode=="value":
        vmin,vmax=float(a01.min()),float(a01.max()); thr=vmax-(vmax-vmin)*(percent/100.0); m=a01>=thr
    else:
        thr=float(np.quantile(a01, 1.0-percent/100.0)); m=a01>=thr
    if dilate>0:
        mm=Image.fromarray((m.astype(np.uint8)*255),"L").filter(ImageFilter.MaxFilter(size=2*dilate+1))
        m=(np.asarray(mm)>0)
    return m

def save_erased_heatmaps(ht_dir, out_dir, mode, percent, alpha, dilate):
    clean(out_dir)
    for g in Path(ht_dir).glob("*.heat.gray.png"):
        a=np.asarray(Image.open(g).convert("L"),dtype=np.float32)/255.0
        m=mask_from_heat(a,mode,percent,alpha,dilate)
        x=(a*255).astype(np.uint8); x[m]=0
        Image.fromarray(x,"L").save(Path(out_dir,g.name))

def apply_masks_to_originals(ht_dir, originals_dir, out_dir, img_size, mode, percent, alpha, dilate, resize_factor=1.14):
    clean(out_dir)
    stems={p.name.replace(".heat.npy","").replace(".heat.gray.png","").replace(".heat.magma.png","").replace(".heat.png","")
           for p in Path(ht_dir).glob("*.heat*")}
    if not stems: return
    for p in imgs(originals_dir):
        if p.stem not in stems: continue
        npy=Path(ht_dir,f"{p.stem}.heat.npy")
        if npy.exists(): a=np.load(npy).astype(np.float32)
        else: a=np.asarray(Image.open(Path(ht_dir,f"{p.stem}.heat.gray.png")).convert("L"),dtype=np.float32)/255.0
        a=(a-a.min())/(a.max()-a.min()+1e-12)
        m=mask_from_heat(a,mode,percent,alpha,dilate)
        im=Image.open(p).convert("RGB"); im=align_like_val(im,img_size,resize_factor)
        arr=np.asarray(im).copy()
        mm=Image.fromarray((m.astype(np.uint8)*255),"L").resize((arr.shape[1],arr.shape[0]), Image.NEAREST)
        arr[(np.asarray(mm)>0)]=0
        Image.fromarray(arr).save(Path(out_dir,p.name))

# --------------------------- training runner ---------------------------
def run_train(train_py, model, pos, data_root, epochs, outdir, img_size, batch, device, resume, workers):
    cmd=[sys.executable, train_py, "--model", model, "--pos", pos, "--dataset-root", str(data_root),
         "--epochs", str(epochs), "--batch-size", str(batch), "--img-size", str(img_size),
         "--outdir", str(outdir), "--device", device, "--num-workers", str(workers)]
    if resume: cmd+=["--resume-weights", str(resume)]
    return os.spawnv(os.P_WAIT, sys.executable, [sys.executable]+cmd[1:])

# --------------------------- main ---------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=["alexnet","resnet"],required=True)
    ap.add_argument("--pos",choices=["cat","dog"],required=True)
    ap.add_argument("--base-dataset-root",required=True)
    ap.add_argument("--original-dataset-root",required=True)
    ap.add_argument("--start-stage",type=int,default=2)
    ap.add_argument("--end-stage",type=int,default=10)
    ap.add_argument("--cycle-epochs",type=int,default=10)
    ap.add_argument("--img-size",type=int,default=256)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--device",choices=["dml","cuda","cpu"],default="dml")     # train on DML/CUDA/CPU
    ap.add_argument("--ht-device",choices=["cpu","dml","cuda"],default="cpu")  # heatmaps on CPU by default
    ap.add_argument("--outdir",default=r"D:\progs\work\train_model")
    ap.add_argument("--num-workers",type=int,default=0)
    ap.add_argument("--resume-weights",default=None)
    ap.add_argument("--layer",default="block4.bn2")              # alexnet: features[20]
    ap.add_argument("--mask-mode",choices=["alpha","rank","value"],default="alpha")
    ap.add_argument("--alpha",type=float,default=0.5)
    ap.add_argument("--percent",type=float,default=20.0)
    ap.add_argument("--dilate",type=int,default=1)
    ap.add_argument("--train-py",default="train.py")
    args=ap.parse_args()

    base,orig = Path(args.base_dataset_root), Path(args.original_dataset_root)
    dev_ht = get_device(args.ht_device)
    resume = Path(args.resume_weights) if args.resume_weights else find_latest(args.outdir, args.model, args.pos)

    for K in range(args.start_stage, args.end_stage+1):
        cur=base/f"stage{K}"

        # 1) train @ stageK
        _=run_train(args.train_py, args.model, args.pos, cur, args.cycle_epochs,
                    args.outdir, args.img_size, args.batch_size, args.device, resume, args.num_workers)
        latest=find_latest(args.outdir, args.model, args.pos)
        resume = latest or resume
        if resume is None:
            raise SystemExit("Нет доступных весов: передай --resume-weights или дождись успешной тренировки.")

        # 2) heatmaps → stageK/ht_after_stK/{cat,not_cat}
        ht_after=cur/f"ht_after_st{K}"
        for cls in [args.pos, f"not_{args.pos}"]:
            make_heatmaps(args.model, resume, args.layer, cur/cls/"train", ht_after/cls, args.img_size, dev_ht)

        # 3) erased heatmaps + apply masks to ORIGINALS(dataset_for_cat) → ht_with_erasing
        ht_erased=cur/f"ht_erased_st{K}"
        ht_with=cur/"ht_with_erasing"
        for cls in [args.pos, f"not_{args.pos}"]:
            save_erased_heatmaps(ht_after/cls, ht_erased/cls, args.mask_mode, args.percent, args.alpha, args.dilate)
            apply_masks_to_originals(ht_after/cls, orig/cls/"train", ht_with/cls,
                                     args.img_size, args.mask_mode, args.percent, args.alpha, args.dilate)

        # 4) собрать следующий стейдж (кроме последнего)
        if K<args.end_stage:
            nxt=base/f"stage{K+1}"
            clean(nxt/args.pos/"train"); clean(nxt/f"not_{args.pos}"/"train")
            shutil.copytree(ht_with/args.pos, nxt/args.pos/"train", dirs_exist_ok=True)
            shutil.copytree(ht_with/f"not_{args.pos}", nxt/f"not_{args.pos}"/"train", dirs_exist_ok=True)
            # val оставляем как есть во всех stage

if __name__=="__main__": main()
