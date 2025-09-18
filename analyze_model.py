# analyze_model.py
"""
CLI-инструмент для быстрого анализа PyTorch-модели:
    • построение графа (если установлен torchviz)
    • вывод количества параметров по каждому тензору
    • суммарные параметры (trainable / frozen)
    • проверка весов на NaN / Inf / слишком большие значения

Примеры:
    # импорт по имени пакета/модуля
    python analyze_model.py --model work.alexnet --img-size 224

    # импорт по пути к файлу .py
    python analyze_model.py --model .\\work\\alexnet.py --img-size 224

Требования: torch, numpy, (опционально) torchviz.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn

try:
    from torchviz import make_dot
except ImportError:  # torchviz не обязателен
    make_dot = None
    warnings.warn("torchviz не найден → визуализация графа будет пропущена")


# ──────────────────────────── CLI ────────────────────────────
parser = argparse.ArgumentParser("Model analyser")
parser.add_argument(
    "--model",
    required=True,
    help="python-модуль ИЛИ путь к файлу .py с моделью (напр., work.alexnet ИЛИ .\\work\\alexnet.py)",
)
parser.add_argument("--img-size", type=int, default=360, help="квадратная сторона dummy-входа")
parser.add_argument("--save-dir", default="model_analysis", help="куда сохранять граф")
args = parser.parse_args()


# ───────────── динамический импорт модуля (имя или путь) ─────────────
def _load_module(name_or_path: str):
    p = Path(name_or_path)

    # 1) Явный файл *.py
    if p.suffix.lower() == ".py" and p.exists():
        spec = importlib.util.spec_from_file_location(p.stem, p)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Не удалось создать spec для файла: {p}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[p.stem] = mod
        spec.loader.exec_module(mod)
        return mod

    # 2) Похоже на путь (есть '/' или '\') — попробуем вывести *.py
    if any(sep in name_or_path for sep in ("/", "\\")):
        if p.is_dir():
            candidate = p / "__init__.py"
            if not candidate.exists():
                py_files = sorted(p.glob("*.py"))
                if py_files:
                    candidate = py_files[0]
            if candidate.exists():
                return _load_module(str(candidate))
        else:
            candidate = p.with_suffix(".py")
            if candidate.exists():
                return _load_module(str(candidate))

    # 3) Обычный импорт по «точечному» имени
    try:
        # на всякий случай добавим текущую папку в sys.path
        sys.path.insert(0, str(Path.cwd()))
        return importlib.import_module(name_or_path)
    except ModuleNotFoundError:
        # 4) Последняя попытка: поиск файла <имя>.py в подкаталогах текущей директории
        hits = list(Path.cwd().rglob(f"{name_or_path}.py"))
        if len(hits) == 1:
            return _load_module(str(hits[0]))
        elif len(hits) > 1:
            raise RuntimeError(
                f"Найдено несколько файлов '{name_or_path}.py':\n"
                + "\n".join(f"  • {h}" for h in hits)
                + "\nУточните путь явно (--model PATH_TO_FILE.py)."
            )
        raise  # пробрасываем исходную ModuleNotFoundError


mod = _load_module(args.model)

# ───────────── поиск класса модели ─────────────
ModelCls: Optional[type[nn.Module]] = None
for alias in ("CNNModel", "Model"):
    if hasattr(mod, alias):
        ModelCls = getattr(mod, alias)
        break

if ModelCls is None:
    # Fallback: первый nn.Module из модуля, который можно вызвать без аргументов
    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if issubclass(cls, nn.Module) and cls.__module__ == mod.__name__:
            try:
                inspect.signature(cls).bind()  # можно вызвать без аргументов
                ModelCls = cls
                break
            except TypeError:
                continue

if ModelCls is None:
    raise RuntimeError(f"Не удалось найти класс-модель в модуле {args.model}")

# ───────────── создаём модель + dummy input ─────────────
device = torch.device("cpu")  # для анализа CPU достаточно
model = ModelCls().to(device).eval()
example_input = torch.randn(1, 3, args.img_size, args.img_size, device=device)

# ───────────── 1. визуализация графа ─────────────
os.makedirs(args.save_dir, exist_ok=True)
if make_dot is not None:
    try:
        out: Any = model(example_input)
        # Если модель вернула не чистый Tensor, попробуем извлечь первый подходящий тензор
        if isinstance(out, (tuple, list)):
            out = next((t for t in out if isinstance(t, torch.Tensor)), out[0])
        elif isinstance(out, dict):
            for k in ("logits", "out", "pred", "y"):
                if k in out and isinstance(out[k], torch.Tensor):
                    out = out[k]
                    break
        if not isinstance(out, torch.Tensor):
            raise TypeError("Ожидался выход типа Tensor для построения графа")

        graph = make_dot(out, params=dict(model.named_parameters()))
        graph.format = "png"
        graph.attr(dpi="300", fontname="Helvetica-Bold", fontsize="12")
        out_path = Path(args.save_dir) / f"{Path(args.model).stem}_graph"
        graph.render(str(out_path))
        print(f"[viz] граф сохранён → {out_path}.png")
    except Exception as e:
        print(f"[viz] ошибка визуализации: {e}")
else:
    print("[viz] torchviz отсутствует — пропускаю граф")

# ───────────── 2. параметры по тензорам + сводка ─────────────
print(f"\n{args.model}.{ModelCls.__name__}\n")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print("[params] параметры по тензорам:")
for name, p in model.named_parameters():
    status = "T" if p.requires_grad else "F"  # Trainable / Frozen
    print(f"  {name:50} : {p.numel():>10,}  ({status})")

print(f"\n  ➤ Всего параметров : {total_params:,}")
print(f"  ➤ Обучаемых       : {trainable_params:,}")
print(f"  ➤ Замороженных    : {frozen_params:,}\n")

# ───────────── 3. проверка весов ─────────────
print("[check] аномалии весов (NaN | Inf | |x|>1e3):")
for name, p in model.named_parameters():
    data = p.detach().cpu().numpy()
    flags = []
    if np.isnan(data).any():
        flags.append("NaN")
    if np.isinf(data).any():
        flags.append("Inf")
    big = int((np.abs(data) > 1e3).sum())
    if big:
        flags.append(f">{big} элементов |x|>1e3")
    status = " | ".join(flags) if flags else "OK"
    print(f"  {name:50} : {p.numel():>10,}  →  {status}")

print("\n✔ Анализ завершён.")
