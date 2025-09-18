#!/usr/bin/env python
# model_summary.py — печать структуры модели и форм активаций (все модули + functional Linear)
from __future__ import annotations
import argparse, importlib, importlib.util, inspect, sys, re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

def human(n: int) -> str:
    return f"{n:,}".replace(",", " ")

def is_leaf(m: nn.Module) -> bool:
    return len(list(m.children())) == 0

def resolve_model_class(mod) -> type[nn.Module]:
    return (getattr(mod, "CNNModel", None)
            or getattr(mod, "Model", None)
            or next(cls for _, cls in inspect.getmembers(mod, inspect.isclass)
                    if issubclass(cls, nn.Module) and cls.__module__ == mod.__name__))

def safe_shape(x: Any) -> str:
    if torch.is_tensor(x):
        return str(tuple(x.shape))
    if isinstance(x, (list, tuple)) and x and all(torch.is_tensor(t) for t in x):
        return "list/tuple[" + ", ".join(str(tuple(t.shape)) for t in x) + "]"
    if isinstance(x, dict):
        keys = ", ".join(map(str, x.keys()))
        return f"dict[{keys}]"
    return f"{type(x).__name__}"

def collect_before_pools(net: nn.Module) -> Dict[str, Tuple[str, nn.Module]]:
    out: Dict[str, Tuple[str, nn.Module]] = {}
    last_conv_name: str | None = None
    last_conv_mod: nn.Module | None = None
    k = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv_name, last_conv_mod = name, m
        if isinstance(m, nn.MaxPool2d) and last_conv_mod is not None:
            k += 1
            out[f"before_pool{k}"] = (last_conv_name, last_conv_mod)
            last_conv_name, last_conv_mod = None, None
    if last_conv_mod is not None and last_conv_name is not None:
        out["last_conv"] = (last_conv_name, last_conv_mod)
    return out

def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def load_module(name_or_path: str):
    p = Path(name_or_path)
    if p.suffix.lower() == ".py" and p.exists():
        spec = importlib.util.spec_from_file_location(p.stem, p)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"spec is None for {p}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[p.stem] = mod
        spec.loader.exec_module(mod)
        return mod
    if any(sep in name_or_path for sep in ("/", "\\")):
        if p.is_dir():
            cand = p / "__init__.py"
            if not cand.exists():
                py = sorted(p.glob("*.py"))
                if py: cand = py[0]
            if cand.exists(): return load_module(str(cand))
        else:
            cand = p.with_suffix(".py")
            if cand.exists(): return load_module(str(cand))
    sys.path.insert(0, str(Path.cwd()))
    try:
        return importlib.import_module(name_or_path)
    except ModuleNotFoundError:
        hits = list(Path.cwd().rglob(f"{name_or_path}.py"))
        if len(hits) == 1:
            return load_module(str(hits[0]))
        if len(hits) > 1:
            raise RuntimeError("Найдено несколько вариантов:\n" + "\n".join(f"  • {h}" for h in hits))
        raise

def main():
    cli = argparse.ArgumentParser("Model summary + activation shapes")
    cli.add_argument("--model", required=True)
    cli.add_argument("--weights", default=None)
    cli.add_argument("--img-size", type=int, default=224)
    cli.add_argument("--device", default="cpu", choices=["cpu","cuda","dml"])
    args = cli.parse_args()

    mod = load_module(args.model)
    Model = resolve_model_class(mod)
    net = Model()

    # загрузка весов
    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu")
        state = None
        if isinstance(ckpt, dict):
            for k in ("state_dict","model","weights"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    state = ckpt[k]; break
        if state is None and isinstance(ckpt, dict):
            state = ckpt
        if state is None:
            raise RuntimeError("Не распознан формат весов (ожидался state_dict)")
        state = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }
        net.load_state_dict(state, strict=False)

    if args.device == "cuda":
        net.to(torch.device("cuda"))
    elif args.device == "dml":
        import torch_directml  # type: ignore
        net.to(torch_directml.device())
    net.eval()

    # таблицы
    shapes: "OrderedDict[str,str]" = OrderedDict()
    types: Dict[str,str] = {}
    params_local: Dict[str,int] = {}
    trainable_local: Dict[str,int] = {}

    # хуки на листьях
    handles = []
    for name, m in net.named_modules():
        types[name] = type(m).__name__
        params_local[name] = sum(p.numel() for p in m.parameters(recurse=False))
        trainable_local[name] = sum(p.numel() for p in m.parameters(recurse=False) if p.requires_grad)
        if is_leaf(m):
            def mk_hook(nm):
                def hook(_m, _in, out):
                    try: shapes[nm] = safe_shape(out)
                    except Exception as e: shapes[nm] = f"<err: {e}>"
                return hook
            handles.append(m.register_forward_hook(mk_hook(name)))

    # пробег через CPU
    with torch.no_grad():
        x = torch.zeros(1, 3, args.img_size, args.img_size)
        _ = net.cpu()(x)
    for h in handles: h.remove()

    # обнаружение «функциональных» Linear на корне (F.linear с параметрами)
    # группируем пары *.weight (2D) и соответствующий *.bias (1D) у корня
    root_params = {n: p for n,p in net.named_parameters() if "." not in n}
    fun_linear_rows = []
    for n, w in root_params.items():
        if not n.endswith("weight"): continue
        if w.ndim != 2: continue
        base = n[:-6]  # отрезать 'weight'
        b = root_params.get(base + "bias", None)
        out_f, in_f = w.shape  # [out_features, in_features]
        pname = base.rstrip(".") if base else "root_fc"
        count = w.numel() + (b.numel() if isinstance(b, torch.Tensor) else 0)
        params_local[pname] = count
        trainable_local[pname] = count
        types[pname] = "Linear (functional)"
        shapes.setdefault(pname, "— (не исполнялся)")  # формы не снимем — нет модуля
        fun_linear_rows.append((pname, out_f, in_f, count))

    # пары conv→pool
    bp = collect_before_pools(net)
    if bp:
        print("\n┌─ conv → MaxPool2d (conv BEFORE pool) ─────────────────")
        for k, (path, _) in bp.items():
            print(f"  {k:>12}: {path}")
        print("└───────────────────────────────────────────────────────\n")

    # печать таблицы
    print("layer (path)                                │ type                    │ out shape                 │ params    │ trainable")
    print("─────────────────────────────────────────────┼─────────────────────────┼───────────────────────────┼───────────┼──────────")
    names_all = list(types.keys())
    names_all.sort(key=natural_key)
    for name in names_all:
        t  = types.get(name, "?")
        p  = human(params_local.get(name, 0))
        tr = human(trainable_local.get(name, 0))
        shape = shapes.get(name, "— (не исполнялся)")
        print(f"{name:41s} │ {t:23s} │ {shape:25s} │ {p:9s} │ {tr:9s}")

    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nTotal params: {human(total)} | Trainable: {human(trainable)} | Frozen: {human(total-trainable)}")

    if fun_linear_rows:
        print("\n[info] Обнаружены «функциональные» Linear на корневом модуле:")
        for nm, out_f, in_f, cnt in fun_linear_rows:
            print(f"  • {nm}: Linear({in_f} → {out_f})  | params={human(cnt)}")

    feats = sorted([n for n in types.keys() if n.startswith("features.")], key=natural_key)
    if feats:
        nums = sorted({int(m.group(1)) for m in map(lambda s: re.search(r"features\.(\d+)", s), feats) if m})
        holes = []
        for a,b in zip(nums, nums[1:]):
            if b - a > 1: holes.append((a+1,b-1))
        if holes:
            print("[note] features.* пропуски: " + "; ".join(f"{a}–{b}" if a!=b else str(a) for a,b in holes))

    print("\n[hint] Путь из первой колонки можно использовать в других скриптах (например, --probe \"features[8]\" или \"block4.bn2\").")

if __name__ == "__main__":
    main()
