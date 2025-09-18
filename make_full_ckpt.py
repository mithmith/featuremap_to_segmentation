#!/usr/bin/env python
"""
make_full_ckpt.py – превращает «weights‑only» .pt в полный чек‑пойнт
Usage:
  python make_full_ckpt.py ^
    --src  alexnet_cat_20250715_100640.pt ^
    --dst  alexnet_cat_20250715_100640_e100.pt ^
    --model alexnet.CNNModel ^
    --epoch 100
"""

import argparse, importlib, torch
import torch.optim as optim

cli = argparse.ArgumentParser()
cli.add_argument("--src",   required=True)
cli.add_argument("--dst",   required=True)
cli.add_argument("--model", required=True, help="module.ClassName")
cli.add_argument("--epoch", required=True, type=int)
args = cli.parse_args()

# 1) создаём сеть
mod_name, cls_name = args.model.rsplit(".", 1)
Model = getattr(importlib.import_module(mod_name), cls_name)
net = Model()                               # ← экземпляр модели

# 2) загружаем веса
try:
    state_dict = torch.load(args.src, map_location="cpu", weights_only=True)
except torch.serialization.pickle.UnpicklingError:
    print("[warn] weights_only=True failed, falling back to full load")
    state_dict = torch.load(args.src, map_location="cpu")
net.load_state_dict(state_dict, strict=False)

# 3) пустой оптимизатор (или задайте реальные hyper‑params)
optimizer = optim.SGD(net.parameters(), lr=1e-3)

# 4) сохраняем «полный» чек‑пойнт
torch.save(
    {"model": net.state_dict(),
     "optim": optimizer.state_dict(),
     "epoch": args.epoch},
    args.dst,
)
print(f"[✓] created {args.dst}")
