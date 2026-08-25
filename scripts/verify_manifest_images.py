from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify every unique image in a manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--full-load",
        action="store_true",
        help="decode pixels exactly as the embedding backend does instead of only checking PNG structure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths: list[Path] = []
    seen: set[str] = set()
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            value = str(row["image"])
            if value not in seen:
                seen.add(value)
                paths.append(Path(value))
    bad: list[dict[str, str]] = []
    for position, path in enumerate(paths, start=1):
        try:
            with Image.open(path) as image:
                if args.full_load:
                    image.convert("RGB").load()
                else:
                    image.verify()
        except Exception as error:  # PIL raises several format-specific exception types.
            bad.append(
                {
                    "image": str(path.resolve()),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(f"bad image: {path}: {type(error).__name__}: {error}", flush=True)
        if position % 5000 == 0 or position == len(paths):
            print(f"verified {position}/{len(paths)}; bad={len(bad)}", flush=True)
    report = {
        "manifest": str(args.manifest.resolve()),
        "mode": "full_load" if args.full_load else "verify",
        "unique_images": len(paths),
        "bad_images": bad,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
