# main.py
import argparse
import subprocess
import sys
import shlex
from pathlib import Path
from datetime import datetime


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def _exe() -> str:
    return sys.executable or "python"


def cmd_train(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        _exe(), "train.py",
        "--model", args.model,
        "--pos", args.pos,
        "--dataset-root", args.dataset_root,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--img-size", str(args.img_size),
        "--outdir", str(outdir),
        "--device", args.device,
    ]
    if args.resume_weights:
        cmd += ["--resume-weights", args.resume_weights]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.weight_decay is not None:
        cmd += ["--weight-decay", str(args.weight_decay)]
    if args.num_workers is not None:
        cmd += ["--num-workers", str(args.num_workers)]

    return _run(cmd)


def cmd_eval(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_prefix = f"{args.model}_{args.pos}_{stamp}"

    cmd = [
        _exe(), "evaluate.py",
        "--model", args.model,
        "--pos", args.pos,
        "--dataset-root", args.dataset_root,
        "--weights", args.weights,
        "--outdir", str(outdir),
        "--prefix", default_prefix,
        "--img-size", str(args.img_size),
        "--device", args.device,
    ]
    if args.threshold is not None:
        cmd += ["--threshold", str(args.threshold)]
    if args.num_workers is not None:
        cmd += ["--num-workers", str(args.num_workers)]

    return _run(cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline entry point for training and evaluating binary classifiers (cat/dog)."
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    pt = sub.add_parser("train", help="Train a model.")
    pt.add_argument("--model", choices=["alexnet", "resnet"], required=True)
    pt.add_argument("--pos", choices=["cat", "dog"], required=True)
    pt.add_argument("--dataset-root", required=True, help=r'Root like "D:\progs\work\datasets\dataset_for_cat"')
    pt.add_argument("--epochs", type=int, required=True)
    pt.add_argument("--batch-size", type=int, default=24)
    pt.add_argument("--img-size", type=int, default=256)
    pt.add_argument("--outdir", default=r"D:\progs\work\train_model")
    pt.add_argument("--device", choices=["dml", "cuda", "cpu"], default="dml")
    pt.add_argument("--resume-weights", default=None, help="Path to .pt state_dict to finetune from.")
    pt.add_argument("--lr", type=float, default=None)
    pt.add_argument("--seed", type=int, default=None)
    pt.add_argument("--weight-decay", type=float, default=None)
    pt.add_argument("--num-workers", type=int, default=None)
    pt.set_defaults(func=cmd_train)

    # ── eval ───────────────────────────────────────────────────────────────
    pe = sub.add_parser("eval", help="Evaluate a trained model.")
    pe.add_argument("--model", choices=["alexnet", "resnet"], required=True)
    pe.add_argument("--pos", choices=["cat", "dog"], required=True)
    pe.add_argument("--dataset-root", required=True, help=r'Root like "D:\progs\work\datasets\dataset_for_cat"')
    pe.add_argument("--weights", required=True, help="Path to saved .pt state_dict.")
    pe.add_argument("--img-size", type=int, default=256)
    pe.add_argument("--outdir", default=r"D:\progs\work\train_model")
    pe.add_argument("--device", choices=["dml", "cuda", "cpu"], default="dml")
    pe.add_argument("--threshold", type=float, default=None)
    pe.add_argument("--num-workers", type=int, default=None)
    pe.set_defaults(func=cmd_eval)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
