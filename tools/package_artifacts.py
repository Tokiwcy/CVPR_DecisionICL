"""Inventory/package the published result evidence, preserving original bytes.

Archives use a SHA-256 object store to avoid copying identical isolated-resume
files repeatedly. The manifest maps original workspace-relative paths to objects.
Dataset images and model checkpoints are explicitly outside this evidence bundle.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import time
import zipfile


SHA = re.compile(r"^[0-9a-f]{64}$")
IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
WEIGHTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
REPO = Path(__file__).resolve().parents[1]
EXPORT_INPUTS = {
    "ledger_method_csv": "docs/experiment_results_ledger_20260920_zh.csv",
    "ledger_paired_csv": "docs/experiment_results_ledger_20260920_zh.paired.csv",
    "ledger_source_manifest": "docs/experiment_results_ledger_20260920_zh.sources.json",
    "paper_tables": "paper/cvpr2027/tables.md",
    "paper_numbers": "paper/cvpr2027/paper_numbers.json",
    "matched_source_control": "paper/cvpr2027/matched_source_control.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def refs(value):
    """Read declared path→SHA maps and explicit path/SHA records only."""
    if isinstance(value, dict):
        for key, digest in value.items():
            if isinstance(digest, str) and SHA.fullmatch(digest) and (
                "/" in key or "\\" in key or Path(key).suffix
            ):
                yield key, digest
        path = value.get("path") or value.get("file") or value.get("source")
        digest = value.get("sha256") or value.get("file_sha256")
        if isinstance(path, str) and isinstance(digest, str) and SHA.fullmatch(digest):
            yield path, digest
        for child in value.values():
            yield from refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from refs(child)


class Inventory:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.clean = self.workspace / "decision_guided_retrieval_cleanroom"
        self.files: dict[str, dict] = {}
        self.expected: dict[str, set[str]] = {}
        self.excluded: dict[str, dict] = {}
        self.missing: list[dict] = []
        self.pending: deque[Path] = deque()
        self.scanned: set[Path] = set()
        self.historical_versions: dict[str, dict] = {}

    def resolve(self, raw: str, parent: Path) -> Path | None:
        normalized = raw.replace("\\", "/")
        win = PureWindowsPath(raw)
        if win.is_absolute():
            path = Path(normalized)
            if path.is_relative_to(self.workspace):
                return path
            return None
        choices = (parent / normalized, self.workspace / normalized, self.clean / normalized)
        return next((p for p in choices if p.is_file()), choices[0])

    def add(self, path: Path, expected: str | None = None, parent: str = "seed"):
        # Lexical normalization avoids repeated Windows reparse-point walks for
        # hundreds of thousands of already enumerated raw files.
        path = Path(os.path.abspath(path))
        if not path.is_relative_to(self.workspace):
            self.excluded[str(path)] = {"reason": "outside_workspace", "sha256": expected}
            return
        relative = path.relative_to(self.workspace).as_posix()
        if path.suffix.lower() in IMAGE | WEIGHTS or "/models/" in f"/{relative}/":
            self.excluded[relative] = {"reason": "dataset_image_or_model_weight", "sha256": expected}
            return
        if not path.is_file():
            self.missing.append({"path": relative, "expected_sha256": expected, "referenced_by": parent})
            return
        if expected:
            self.expected.setdefault(relative, set()).add(expected)
        if relative not in self.files:
            category = "numeric-resources" if path.suffix.lower() in {".npz", ".npy"} else (
                "raw-and-cache" if any("raw" in part or "batches" in part for part in path.parts)
                or "cache" in path.name.lower() else "audit-and-predictions"
            )
            self.files[relative] = {"size": path.stat().st_size, "category": category}
            if path.suffix.lower() == ".json":
                self.pending.append(path)

    def add_block(self, folder: Path):
        for path in folder.iterdir():
            if path.is_file() and path.suffix in {".json", ".jsonl", ".md", ".log"}:
                # Only the completed block, not adjacent incomplete experiments.
                self.add(path)
        for name in ("raw_batches", "raw_descriptions"):
            if (folder / name).is_dir():
                for path in (folder / name).rglob("*"):
                    if path.is_file():
                        self.add(path)
        print(json.dumps({"stage": "block_collected", "folder": folder.relative_to(self.workspace).as_posix(), "files": len(self.files)}), flush=True)

    def collect(self):
        source = json.loads((REPO / "results/source_manifest.json").read_text())
        export = json.loads((REPO / "results/export_manifest.json").read_text())
        for key, relative in EXPORT_INPUTS.items():
            self.add(self.workspace / relative, export["source_sha256"][key])
        for relative, digest in source["sources_sha256"].items():
            path = self.workspace / relative
            self.add(path, digest)
            if path.name == "complete.json" and path.parent.joinpath("predictions.json").exists():
                self.add_block(path.parent)

        # Qwen main live tables point to completed execution roots separately.
        for family in ("qwen3_five_dataset_recalibrated_20260910", "qwen25_nf4_five_dataset_20260913"):
            locations = self.clean / "artifacts" / family / "execution_locations.json"
            self.add(locations)
            for key, raw in json.loads(locations.read_text()).items():
                if key.startswith("_"):
                    continue
                folder = Path(raw)
                if not (folder / "complete.json").is_file():
                    raise ValueError(f"main block is incomplete: {key}: {folder}")
                self.add_block(folder)

        # Include the four same-architecture source-control receipts exactly.
        for relative, digest in source["sources_sha256"].items():
            if "decision_generic" in relative:
                self.add(self.workspace / relative, digest)

        matched = json.loads((REPO / "results/matched_source_control.json").read_text())
        for dataset, folder in (("dtd", "qwen3_core_sources/dtd"), ("aircraft", "qwen3_decision_generic/aircraft")):
            block = self.clean / "artifacts/priority_1shot_ablation_20260920" / folder
            row = next(item for item in matched if item["dataset"] == dataset)
            self.add(block / "complete.json", row["complete_sha256"])
            self.add(block / "independent_audit.json", row["independent_audit_sha256"])
            self.add(block.parent / "run.freeze.json")
            self.add_block(block)
        self.add(self.workspace / "paper/cvpr2027/verify_matched_source_control.py")

        while self.pending:
            path = self.pending.popleft()
            if path in self.scanned:
                continue
            self.scanned.add(path)
            try:
                value = json.loads(path.read_text(encoding="utf-8-sig"))
            except (ValueError, UnicodeError):
                continue
            for raw, digest in refs(value):
                resolved = self.resolve(raw, path.parent)
                if resolved is None:
                    self.excluded[raw] = {"reason": "outside_workspace", "sha256": digest}
                else:
                    self.add(resolved, digest, path.relative_to(self.workspace).as_posix())
            if len(self.scanned) % 10000 == 0:
                print(json.dumps({"stage": "inventory", "files": len(self.files), "scanned_json": len(self.scanned)}), flush=True)
        return source

    def verify(self, output, reuse_verified=False):
        mismatches = []
        started = time.monotonic()
        items = sorted(self.files.items())
        def calculate(item):
            relative, info = item
            return info["sha256"] if reuse_verified and "sha256" in info else sha256(self.workspace / relative)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for index, ((relative, info), digest) in enumerate(zip(items, pool.map(calculate, items)), 1):
                info["sha256"] = digest
                declared = self.expected.get(relative, set())
                info["declared_sha256"] = sorted(declared)
                for old in declared - {digest}:
                    path = self.workspace / relative
                    recovered = False
                    if path.suffix == ".jsonl":
                        payload = path.read_bytes()
                        h, end = hashlib.sha256(), 0
                        for line in payload.splitlines(keepends=True):
                            h.update(line)
                            end += len(line)
                            if h.hexdigest() == old:
                                destination = output / "recovered_objects" / old
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                destination.write_bytes(payload[:end])
                                self.historical_versions.setdefault(relative, {})[old] = {
                                    "sha256": old, "size": end, "recovery": "exact_byte_prefix_of_final_cache",
                                    "source_path": str(destination),
                                }
                                recovered = True
                                break
                    if not recovered:
                        mismatches.append({"path": relative, "actual_sha256": digest, "declared_sha256": old})
                if index % 10000 == 0:
                    print(json.dumps({"stage": "hashing", "verified": index, "total": len(self.files), "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        return mismatches


def pack(inventory, source, output, max_part_mib, base=None):
    objects = {}
    rank = {"audit-and-predictions": 0, "raw-and-cache": 1, "numeric-resources": 2}
    for relative, info in sorted(inventory.files.items()):
        obj = objects.setdefault(info["sha256"], {"path": relative, "source_path": str(inventory.workspace / relative), "size": info["size"], "category": info["category"]})
        if "archive" in info:
            obj["archive"], obj["member"] = info["archive"], info["member"]
        if rank[info["category"]] < rank[obj["category"]]:
            obj["category"] = info["category"]
    for relative, versions in inventory.historical_versions.items():
        for digest, version in versions.items():
            objects.setdefault(digest, {"path": relative + " [historical prefix]", "source_path": version["source_path"], "size": version["size"], "category": "raw-and-cache"})
    assets = list(base["archive_assets"]) if base else []
    for category in rank:
        groups, current, size = [], [], 0
        for digest, info in sorted(objects.items()):
            if info["category"] != category or "archive" in info:
                continue
            if current and size + info["size"] > max_part_mib * 2**20:
                groups.append(current)
                current, size = [], 0
            current.append((digest, info))
            size += info["size"]
        if current:
            groups.append(current)
        for part, group in enumerate(groups, 1):
            prefix = "decisionicl-supplement" if base else "decisionicl"
            name = f"{prefix}-{category}-{part:03d}.zip"
            destination = output / name
            if destination.exists():
                raise FileExistsError(f"Archive already exists; keep the prior package: {destination}")
            print(json.dumps({"stage": "packing", "name": name, "objects": len(group)}), flush=True)
            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
                for index, (digest, info) in enumerate(group, 1):
                    member = "objects/" + digest
                    entry = zipfile.ZipInfo(member, date_time=(2026, 10, 9, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    entry.external_attr = 0o644 << 16
                    actual = hashlib.sha256()
                    with Path(info["source_path"]).open("rb") as src, archive.open(entry, "w", force_zip64=True) as dst:
                        for block in iter(lambda: src.read(4 * 1024**2), b""):
                            actual.update(block)
                            dst.write(block)
                    if actual.hexdigest() != digest:
                        raise ValueError("File changed during packing: " + info["path"])
                    info["archive"] = name
                    info["member"] = member
                    if index % 10000 == 0:
                        print(json.dumps({"stage": "packing", "name": name, "objects_written": index}), flush=True)
            asset = {"name": name, "size": destination.stat().st_size, "sha256": sha256(destination), "category": category}
            if asset["size"] >= 2 * 1024**3:
                raise ValueError("Archive exceeds the GitHub release asset limit: " + name)
            assets.append(asset)
            print(json.dumps({"stage": "archive_ready", **asset}), flush=True)
    for info in inventory.files.values():
        obj = objects[info["sha256"]]
        info["archive"], info["member"] = obj["archive"], obj["member"]
    history = {}
    for relative, versions in inventory.historical_versions.items():
        history[relative] = {}
        for digest, version in versions.items():
            history[relative][digest] = {k: v for k, v in version.items() if k != "source_path"}
            history[relative][digest].update(archive=objects[digest]["archive"], member=objects[digest]["member"])
    manifest = {"format": "decisionicl-evidence-cas-v1", "source_manifest_sha256": sha256(REPO / "results/source_manifest.json"),
        "source_seed_count": len(source["sources_sha256"]), "original_workspace": "D:/BehaviorICL", "files": inventory.files,
        "excluded": inventory.excluded, "archive_assets": assets, "unique_objects": len(objects), "historical_versions": history}
    path = output / "decisionicl-artifact-manifest.json.gz"
    with path.open("wb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        compressed.write(json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode())
    assets.append({"name": path.name, "size": path.stat().st_size, "sha256": sha256(path), "category": "index"})
    # Small, ordinary-path archive for collaborators who only need table review.
    review_paths = set(source["sources_sha256"]) | set(EXPORT_INPUTS.values())
    evaluation = json.loads((REPO / "artifact_bundle/evaluation_index.json").read_text())
    for row in evaluation["methods"]:
        review_paths.update((row["predictions"], row["test_manifest"]))
        parent = Path(row["predictions"]).parent
        for name in ("results.json", "complete.json", "independent_audit.json", "independent_plan_audit.json", "calibration.json", "inference_protocol.json", "plan.freeze.json", "test_plan.jsonl"):
            relative = (parent / name).as_posix()
            if relative in inventory.files:
                review_paths.add(relative)
    for folder in ("qwen3_core_sources/dtd", "qwen3_decision_generic/aircraft"):
        base_path = "decision_guided_retrieval_cleanroom/artifacts/priority_1shot_ablation_20260920/" + folder
        for name in ("results.json", "complete.json", "independent_audit.json", "predictions.json"):
            review_paths.add(base_path + "/" + name)
    review = output / "decisionicl-review-artifacts.zip"
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in sorted(review_paths):
            archive.write(inventory.workspace / relative, relative)
    review_info = {"name": review.name, "size": review.stat().st_size, "sha256": sha256(review), "category": "review"}
    assets.append(review_info)
    review_index = {"archive": review_info, "files": {p: {"sha256": inventory.files[p]["sha256"], "size": inventory.files[p]["size"]} for p in sorted(review_paths)}}
    (output / "review_manifest.json").write_text(json.dumps(review_index, ensure_ascii=True, indent=2) + "\n")
    (output / "SHA256SUMS.txt").write_text("".join(f"{a['sha256']}  {a['name']}\n" for a in assets), encoding="utf-8")
    sums = output / "SHA256SUMS.txt"
    assets.append({"name": sums.name, "size": sums.stat().st_size, "sha256": sha256(sums), "category": "index"})
    body = ("Audited evidence for DecisionICL: the 180 main method results, 34 main controls, four matched factor-source comparisons, "
        "and the separately marked 41 audited interim seed-repeat results. Original files are preserved byte-for-byte, "
        "with SHA-256 deduplication and an original-path index. Includes predictions, plans, calibration, audit receipts, "
        "raw/cache provenance, source snapshots, metadata and referenced numeric resources. "
        "Dataset images and model checkpoints are external inputs listed in the manifest. "
        "Use tools/verify_artifact_bundle.py from the repository to verify or materialize the downloaded files. "
        "The smaller decisionicl-review-artifacts.zip contains direct sources, prediction maps, test-label manifests and block receipts for table review. "
        "The multi-seed repeat is paused and has no complete three-seed aggregate.")
    (output / "release_assets.json").write_text(json.dumps({"assets": assets, "release_body": body}, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps({"stage": "package_complete", "logical_files": len(inventory.files), "unique_objects": len(objects), "compressed_bytes": sum(a['size'] for a in assets)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--max-part-mib", type=int, default=768)
    parser.add_argument("--base-manifest", type=Path, help="extend an already verified local bundle")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = Inventory(args.workspace)
    base = None
    if args.base_manifest:
        with gzip.open(args.base_manifest, "rt", encoding="utf-8") as stream:
            base = json.load(stream)
        for asset in base["archive_assets"]:
            if sha256(output / asset["name"]) != asset["sha256"]:
                raise ValueError("Base archive changed: " + asset["name"])
        inventory.files = base["files"]
        inventory.expected = {p: set(i["declared_sha256"]) for p, i in inventory.files.items()}
        inventory.excluded = base["excluded"]
    source = inventory.collect()
    sizes = Counter()
    counts = Counter()
    for info in inventory.files.values():
        sizes[info["category"]] += info["size"]
        counts[info["category"]] += 1
    report = {"files": len(inventory.files), "bytes": sum(sizes.values()), "category_files": dict(counts),
              "category_mib": {k: round(v / 2**20, 2) for k, v in sizes.items()},
              "excluded_count": len(inventory.excluded), "missing_count": len(inventory.missing)}
    (output / "inventory.json").write_text(json.dumps({"summary": report, "files": inventory.files,
        "excluded": inventory.excluded, "missing": inventory.missing}, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not args.build:
        return
    mismatches = inventory.verify(output, reuse_verified=bool(base))
    (output / "hash_diagnostics.json").write_text(json.dumps({"mismatches": mismatches, "missing": inventory.missing}, ensure_ascii=True, indent=2) + "\n")
    if mismatches or inventory.missing:
        raise ValueError(f"dependency closure needs review: {len(mismatches)} hash mismatches, {len(inventory.missing)} missing references")
    print(json.dumps({"stage": "verified", "files": len(inventory.files)}), flush=True)
    pack(inventory, source, output, args.max_part_mib, base)


if __name__ == "__main__":
    main()
