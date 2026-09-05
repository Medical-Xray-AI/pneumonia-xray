"""Download and validate the approved Kaggle dataset on this computer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.download import (  # noqa: E402
    DATASET_HANDLE,
    authenticate_kaggle,
    ensure_local_dataset,
    update_env_data_root,
    validate_dataset_inventory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "raw" / "kaggle",
        help="Ignored local destination used when XRAY_DATA_ROOT is not already valid.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Validate an existing extracted dataset instead of downloading.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPOSITORY_ROOT / ".env",
        help="Ignored local environment file updated with XRAY_DATA_ROOT.",
    )
    parser.add_argument("--no-write-env", action="store_true")
    parser.add_argument(
        "--login",
        action="store_true",
        help="Run KaggleHub's interactive login before checking/downloading data.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.login:
        authenticate_kaggle()
    root, downloaded_now = ensure_local_dataset(
        download_dir=args.download_dir,
        data_root=args.data_root,
    )
    inventory = validate_dataset_inventory(root)
    env_path = None
    if not args.no_write_env:
        env_path = update_env_data_root(root, args.env_file)
    print(
        json.dumps(
            {
                "dataset_handle": DATASET_HANDLE,
                "dataset_root": str(root),
                "downloaded_now": downloaded_now,
                "image_count": sum(inventory.values()),
                "env_file_updated": str(env_path) if env_path else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
