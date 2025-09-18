#!/usr/bin/env python
# model_summary.py — печать структуры модели и форм активаций (все модули + functional Linear)
from __future__ import annotations
import argparse, importlib, importlib.util, inspect, sys, re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import torch  # type: ignore
import torch.nn as nn  # type: ignore

def human(n: int) -> str:
    """Format integer with spaces as thousands separators.

    Example: 1234567 -> "1 234 567"
    """
    return f"{n:,}".replace(",", " ")

def natural_key(s: str):
    """Return a key list for natural sorting (numbers as integers)."""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def load_module(name_or_path: str):
    """Load a Python module by name or filesystem path.

    Supports:
    - Direct module name import
    - Path to .py file
    - Package directory (with __init__.py)
    - Best-effort search in current working tree by filename
    """
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
    """CLI entrypoint: prints model modules, activation shapes, and parameter stats."""
    parser = argparse.ArgumentParser("Model summary + activation shapes")
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "dml"])
    args = parser.parse_args()

    # Load model class and instantiate
    model_module = load_module(args.model)
    ModelClass = (
        getattr(model_module, "CNNModel", None)
        or getattr(model_module, "Model", None)
        or next(
            cls
            for _, cls in inspect.getmembers(model_module, inspect.isclass)
            if issubclass(cls, nn.Module) and cls.__module__ == model_module.__name__
        )
    )
    model: nn.Module = ModelClass()

    # Load weights if provided
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu")
        state_dict = None
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "weights"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    state_dict = value
                    break
            if state_dict is None:
                state_dict = checkpoint
        if state_dict is None:
            raise RuntimeError("Не распознан формат весов (ожидался state_dict)")
        state_dict = { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }
        model.load_state_dict(state_dict, strict=False)

    # Move to device
    if args.device == "cuda":
        model.to(torch.device("cuda"))
    elif args.device == "dml":
        import torch_directml  # type: ignore
        model.to(torch_directml.device())
    model.eval()

    # Collect per-module meta and register hooks on leaf modules
    output_shape_by_name: "OrderedDict[str, str]" = OrderedDict()
    module_type_by_name: Dict[str, str] = {}
    param_count_local_by_name: Dict[str, int] = {}
    trainable_param_count_by_name: Dict[str, int] = {}

    forward_hook_handles = []
    for module_path, module in model.named_modules():
        module_type_by_name[module_path] = type(module).__name__
        param_count_local_by_name[module_path] = sum(p.numel() for p in module.parameters(recurse=False))
        trainable_param_count_by_name[module_path] = sum(
            p.numel() for p in module.parameters(recurse=False) if p.requires_grad
        )

        has_children = any(True for _ in module.children())
        if not has_children:
            def make_hook(path: str):
                def hook(_m, _in, out):
                    try:
                        if torch.is_tensor(out):
                            output_shape_by_name[path] = str(tuple(out.shape))
                        elif isinstance(out, (list, tuple)) and out and all(torch.is_tensor(t) for t in out):
                            output_shape_by_name[path] = "list/tuple[" + ", ".join(str(tuple(t.shape)) for t in out) + "]"
                        elif isinstance(out, dict):
                            keys = ", ".join(map(str, out.keys()))
                            output_shape_by_name[path] = f"dict[{keys}]"
                        else:
                            output_shape_by_name[path] = f"{type(out).__name__}"
                    except Exception as e:
                        output_shape_by_name[path] = f"<err: {e}>"
                return hook
            forward_hook_handles.append(module.register_forward_hook(make_hook(module_path)))

    # Single CPU pass to capture activation shapes
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, args.img_size, args.img_size)
        _ = model.cpu()(dummy_input)
    for h in forward_hook_handles:
        h.remove()

    # Detect root-level functional Linear (F.linear) parameter pairs
    root_params = {name: p for name, p in model.named_parameters() if "." not in name}
    root_functional_linear_rows = []
    for param_name, weight in root_params.items():
        if not param_name.endswith("weight"):
            continue
        if weight.ndim != 2:
            continue
        base = param_name[:-6]  # strip 'weight'
        bias_param = root_params.get(base + "bias", None)
        out_features, in_features = weight.shape
        printable_name = base.rstrip(".") if base else "root_fc"
        if bias_param is None:
            bias_count = 0
        else:
            bias_count = int(bias_param.numel())
        param_count = weight.numel() + bias_count
        param_count_local_by_name[printable_name] = param_count
        trainable_param_count_by_name[printable_name] = param_count
        module_type_by_name[printable_name] = "Linear (functional)"
        output_shape_by_name.setdefault(printable_name, "— (не исполнялся)")
        root_functional_linear_rows.append((printable_name, out_features, in_features, param_count))

    # conv → MaxPool2d pairs (conv BEFORE pool)
    last_conv_path: str | None = None
    last_conv_mod: nn.Module | None = None
    conv_before_pools: Dict[str, Tuple[str, nn.Module]] = {}
    counter = 0
    for module_path, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_path, last_conv_mod = module_path, module
        if isinstance(module, nn.MaxPool2d) and last_conv_mod is not None and last_conv_path is not None:
            counter += 1
            conv_before_pools[f"before_pool{counter}"] = (last_conv_path, last_conv_mod)
            last_conv_path, last_conv_mod = None, None
    if last_conv_mod is not None and last_conv_path is not None:
        conv_before_pools["last_conv"] = (last_conv_path, last_conv_mod)
    if conv_before_pools:
        print("\n┌─ conv → MaxPool2d (conv BEFORE pool) ─────────────────")
        for key, (path, _mod) in conv_before_pools.items():
            print(f"  {key:>12}: {path}")
        print("└───────────────────────────────────────────────────────\n")

    # Print table
    print("layer (path)                                │ type                    │ out shape                 │ params    │ trainable")
    print("─────────────────────────────────────────────┼─────────────────────────┼───────────────────────────┼───────────┼──────────")
    all_names = list(module_type_by_name.keys())
    all_names.sort(key=natural_key)
    for name in all_names:
        type_name = module_type_by_name.get(name, "?")
        params_str = human(param_count_local_by_name.get(name, 0))
        trainable_str = human(trainable_param_count_by_name.get(name, 0))
        shape_str = output_shape_by_name.get(name, "— (не исполнялся)")
        print(f"{name:41s} │ {type_name:23s} │ {shape_str:25s} │ {params_str:9s} │ {trainable_str:9s}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {human(total_params)} | Trainable: {human(trainable_params)} | Frozen: {human(total_params - trainable_params)}")

    if root_functional_linear_rows:
        print("\n[info] Обнаружены «функциональные» Linear на корневом модуле:")
        for nm, out_f, in_f, cnt in root_functional_linear_rows:
            print(f"  • {nm}: Linear({in_f} → {out_f})  | params={human(cnt)}")

    feature_path_names = sorted([n for n in module_type_by_name.keys() if n.startswith("features.")], key=natural_key)
    if feature_path_names:
        indices = sorted({int(m.group(1)) for m in map(lambda s: re.search(r"features\.(\d+)", s), feature_path_names) if m})
        gaps: list[Tuple[int, int]] = []
        for a, b in zip(indices, indices[1:]):
            if b - a > 1:
                gaps.append((a + 1, b - 1))
        if gaps:
            print("[note] features.* пропуски: " + "; ".join(f"{a}–{b}" if a != b else str(a) for a, b in gaps))

    print("\n[hint] Путь из первой колонки можно использовать в других скриптах (например, --probe \"features[8]\" или \"block4.bn2\").")

if __name__ == "__main__":
    main()
