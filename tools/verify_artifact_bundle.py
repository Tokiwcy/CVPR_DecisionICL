"""Download/check/materialize the DecisionICL evidence release (stdlib only)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.parse
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True, help="downloaded release directory")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--materialize", type=Path, help="new directory for original logical paths")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    release = json.loads((ROOT / "artifact_bundle/release.json").read_text(encoding="utf-8"))
    for asset in release["assets"]:
        name = asset["name"]
        if Path(name).name != name:
            raise ValueError("Unsafe asset filename")
        path = bundle / name
        if not path.exists() and args.download:
            url = asset["browser_download_url"]
            if urllib.parse.urlparse(url).hostname != "github.com":
                raise ValueError("Unexpected download host")
            print(json.dumps({"stage": "download", "name": name}), flush=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=600) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out, 4 * 1024**2)
            temporary.replace(path)
        expected = asset["digest"].removeprefix("sha256:")
        if not path.is_file() or path.stat().st_size != asset["size"] or digest(path) != expected:
            raise ValueError("Release asset is missing or differs: " + name)
    with gzip.open(bundle / "decisionicl-artifact-manifest.json.gz", "rt", encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["format"] == "decisionicl-evidence-cas-v1"
    repo_source = ROOT / "results/source_manifest.json"
    assert digest(repo_source) == manifest["source_manifest_sha256"]
    declared = json.loads(repo_source.read_text())["sources_sha256"]
    for path, expected in declared.items():
        assert manifest["files"][path]["sha256"] == expected, path

    objects = {}
    for path, info in manifest["files"].items():
        logical = PurePosixPath(path)
        if logical.is_absolute() or ".." in logical.parts or ":" in path:
            raise ValueError("Unsafe logical artifact path")
        historical = manifest.get("historical_versions", {}).get(path, {})
        if any(d != info["sha256"] and d not in historical for d in info["declared_sha256"]):
            raise ValueError("Original declared file hash differs: " + path)
        objects[info["sha256"]] = info
    for versions in manifest.get("historical_versions", {}).values():
        for expected, info in versions.items():
            objects[expected] = info

    target = args.materialize.resolve() if args.materialize else None
    if target and target.exists() and any(target.iterdir()):
        raise ValueError("Materialization requires a new or empty directory")
    if target:
        target.mkdir(parents=True, exist_ok=True)
    archives = {}
    try:
        for index, (expected, info) in enumerate(sorted(objects.items()), 1):
            archive = archives.setdefault(info["archive"], zipfile.ZipFile(bundle / info["archive"])) if info["archive"] not in archives else archives[info["archive"]]
            if info["member"] != "objects/" + expected:
                raise ValueError("Unsafe object member")
            h = hashlib.sha256()
            destination = target / ".objects" / expected if target else None
            if destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info["member"]) as stream:
                if destination:
                    with destination.open("wb") as out:
                        for block in iter(lambda: stream.read(4 * 1024**2), b""):
                            h.update(block)
                            out.write(block)
                else:
                    for block in iter(lambda: stream.read(4 * 1024**2), b""):
                        h.update(block)
            assert h.hexdigest() == expected, info["member"]
            if index % 10000 == 0:
                print(json.dumps({"stage": "verify_objects", "verified": index, "total": len(objects)}), flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    if target:
        for relative, info in manifest["files"].items():
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target / ".objects" / info["sha256"], path)
        (target / "HISTORICAL_VERSIONS.json").write_text(json.dumps(manifest.get("historical_versions", {}), ensure_ascii=True, indent=2) + "\n")
        print(json.dumps({"stage": "materialized", "workspace": str(target)}), flush=True)
    print(json.dumps({"status": "passed", "logical_files": len(manifest["files"]),
        "unique_objects": len(objects), "direct_ledger_sources_verified": len(declared),
        "external_input_entries": len(manifest["excluded"])}), flush=True)


if __name__ == "__main__":
    main()
