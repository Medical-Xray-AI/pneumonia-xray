"""End-to-end entry point for pediatric pneumonia classification.

Subcommands
-----------
  check           Validate configs, split_v1 manifests, environment and torch.
  train           Train one config (scripts/train.py): best/last checkpoints,
                  history, predictions_val.csv, metrics.json.
  develop         Full development run: check -> train baseline and DenseNet121
                  -> validation comparison and threshold freeze. Re-running with
                  the same run ids reuses finished runs and resumes broken ones.
  evaluate        Evaluation CLI (scripts/evaluate_model.py): `validation` or `test`.
  infer           Checkpoint inference on image files or a whole split manifest.
  benchmark       Parameter count, model size and forward latency of a checkpoint.
  verify-release  Release verification (scripts/verify_release.py).

Examples
--------
  python run_all.py check --require-data
  python run_all.py develop --run-ids baseline_s42 densenet121_s42 --update-registry
  python run_all.py infer --checkpoint $XRAY_OUTPUT_ROOT/densenet121_s42/checkpoints/best.pt \\
      --split test --threshold-file report/tables/frozen_threshold.json
  python run_all.py infer --checkpoint <best.pt> --threshold-file <frozen.json> --image x.jpeg

The locked test split is predicted only with a validation frozen_threshold.json
that names the same model and run, and it is never part of `develop`.
Settings are read from shell variables or the local .env
(XRAY_DATA_ROOT, XRAY_OUTPUT_ROOT, XRAY_NUM_WORKERS).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import pipeline  # noqa: E402  (lightweight: no torch import at module load)

DEFAULT_CONFIG = Path("configs/densenet121.yaml")
FORWARDED = {"evaluate", "verify-release"}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without importing training dependencies."""
    parser = argparse.ArgumentParser(
        description="Run the pediatric chest X-ray pneumonia pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--check", action="store_true",
                        help="Deprecated alias for `run_all.py check`.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Config for the deprecated --check alias.")
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check", help="Validate configs, manifests and environment.")
    check.add_argument("--config", dest="configs", type=Path, action="append",
                       help="Config to validate (repeatable; default: baseline and densenet121).")
    check.add_argument("--require-data", action="store_true",
                       help="Also require XRAY_DATA_ROOT/XRAY_OUTPUT_ROOT and a valid dataset root.")

    train = sub.add_parser("train", help="Train one configuration.")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--run-id", default=None)
    train.add_argument("--device", default=None, help="auto, cpu or cuda[:N]")
    train.add_argument("--resume", type=Path, default=None, help="<run>/checkpoints/last.pt")
    train.add_argument("--update-registry", action="store_true")

    develop = sub.add_parser("develop", help="Train all configs and freeze on validation.")
    develop.add_argument("--config", dest="configs", type=Path, action="append",
                         help="Config to train (repeatable; default: baseline and densenet121).")
    develop.add_argument("--run-ids", nargs="+", default=None,
                         help="One run id per config, in order; enables reuse/resume.")
    develop.add_argument("--device", default=None)
    develop.add_argument("--out-dir", type=Path, default=pipeline.DEFAULT_REPORT_DIR)
    develop.add_argument("--update-registry", action="store_true")

    sub.add_parser("evaluate", add_help=False,
                   help="Forward to scripts/evaluate_model.py (validation | test).")

    infer = sub.add_parser("infer", help="Predict image files or a split manifest.")
    infer.add_argument("--checkpoint", type=Path, required=True, help="<run>/checkpoints/best.pt")
    target = infer.add_mutually_exclusive_group(required=True)
    target.add_argument("--image", nargs="+", type=Path, help="One or more image files.")
    target.add_argument("--split", choices=["validation", "test"], help="Canonical manifest split.")
    infer.add_argument("--threshold-file", type=Path, default=None,
                       help="Validation frozen_threshold.json (required for --split test).")
    infer.add_argument("--manifest", type=Path, default=None,
                       help="Override data/manifests/<split>.csv.")
    infer.add_argument("--output", type=Path, default=None,
                       help="CSV path (default: <run>/predictions_<split>.csv; stdout for --image).")
    infer.add_argument("--overwrite", action="store_true",
                       help="Replace an existing prediction file.")
    infer.add_argument("--device", default="auto")
    infer.add_argument("--batch-size", type=int, default=32)
    infer.add_argument("--num-workers", type=int, default=None)

    bench = sub.add_parser("benchmark", help="Model size and inference latency.")
    bench.add_argument("--checkpoint", type=Path, required=True)
    bench.add_argument("--device", default="auto")
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument("--repeats", type=int, default=20)
    bench.add_argument("--output", type=Path, default=None, help="Also write the JSON here.")

    sub.add_parser("verify-release", add_help=False,
                   help="Forward to scripts/verify_release.py.")
    return parser


def print_checks(results) -> None:
    for result in results:
        status = "ok" if result.ok else ("FAIL" if result.required else "warn")
        print(f"  [{status:>4}] {result.name}: {result.detail}")


def cmd_check(configs: Optional[List[Path]], require_data: bool) -> int:
    results = pipeline.check_environment(configs or pipeline.DEFAULT_CONFIGS, require_data=require_data)
    print_checks(results)
    if pipeline.checks_passed(results):
        print("Repository entry-point check passed.")
        return 0
    print("Repository entry-point check FAILED.")
    return 1


def cmd_infer(args: argparse.Namespace) -> int:
    if args.image:
        records = pipeline.infer_images(args.checkpoint, args.image, threshold_file=args.threshold_file,
                                        device=args.device, batch_size=args.batch_size)
        from src.inference.schema import records_to_frame
        frame = records_to_frame(records)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
            print(f"[infer] {len(frame)} predictions -> {args.output}")
        else:
            print(json.dumps([r.to_dict() for r in records], indent=2))
        return 0
    pipeline.infer_manifest(args.checkpoint, args.split, threshold_file=args.threshold_file,
                            manifest=args.manifest, output=args.output, overwrite=args.overwrite,
                            device=args.device, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from src.inference import predict

    loaded = predict.load_model(args.checkpoint, device=args.device)
    summary = predict.benchmark(loaded, repeats=args.repeats, batch_size=args.batch_size)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pipeline.load_env_file()

    # Pass-through subcommands keep their own parsers and help text.
    if argv and argv[0] in FORWARDED:
        if argv[0] == "evaluate":
            return pipeline.evaluate(argv[1:])
        from scripts import verify_release
        return verify_release.main(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            if args.check:
                config = args.config or DEFAULT_CONFIG
                if not config.is_file():
                    parser.error(f"Config file not found: {config}")
                return cmd_check([config], require_data=False)
            parser.print_help()
            return 2
        if args.command == "check":
            return cmd_check(args.configs, args.require_data)
        if args.command == "train":
            run_dir = pipeline.train(args.config, run_id=args.run_id, device=args.device,
                                     resume=args.resume, update_registry=args.update_registry)
            print(f"[train] run directory: {run_dir}")
            return 0
        if args.command == "develop":
            pipeline.develop(args.configs or pipeline.DEFAULT_CONFIGS, run_ids=args.run_ids,
                             device=args.device, out_dir=args.out_dir,
                             update_registry=args.update_registry)
            return 0
        if args.command == "infer":
            return cmd_infer(args)
        if args.command == "benchmark":
            return cmd_benchmark(args)
    except (pipeline.PipelineError, FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
