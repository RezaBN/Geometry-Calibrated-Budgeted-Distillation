#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gcbd_vision.config import METHODS, load_config
from gcbd_vision.engine import run_method, run_teacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one GCBD vision teacher or method experiment on one Trillium H100"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("teacher", "method"), required=True)
    parser.add_argument("--dataset", choices=("CIFAR100", "ImageNet"), required=True)
    parser.add_argument("--scenario", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--teacher-epochs", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--clear-stale-lock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.mode == "teacher":
        if args.method is not None:
            raise SystemExit("--method is valid only with --mode method")
        if args.epochs is not None:
            raise SystemExit("Use --teacher-epochs, not --epochs, for a teacher run")
        run_teacher(
            cfg,
            args.dataset,
            args.scenario,
            args.seed,
            epochs=args.teacher_epochs,
            overwrite=args.overwrite,
            resume=not args.no_resume,
            clear_stale_lock=args.clear_stale_lock,
            dry_run=args.dry_run,
        )
    else:
        if args.method is None:
            raise SystemExit("--method is required with --mode method")
        if args.teacher_epochs is not None:
            raise SystemExit("--teacher-epochs is valid only with --mode teacher")
        run_method(
            cfg,
            args.dataset,
            args.scenario,
            args.method,
            args.seed,
            budget=args.budget,
            epochs=args.epochs,
            output_root=args.output_root,
            overwrite=args.overwrite,
            resume=not args.no_resume,
            clear_stale_lock=args.clear_stale_lock,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
