"""Publish a prepared evidence bundle as GitHub release assets.

Uses the existing Git credential helper in memory; credentials are never printed
or written to the package. Existing assets are never overwritten silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request


REPO = "Tokiwcy/CVPR_DecisionICL"
API = "https://api.github.com/repos/" + REPO


def credentials() -> str:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    response = subprocess.run(["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n", text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False)
    values = dict(line.split("=", 1) for line in response.stdout.splitlines() if "=" in line)
    if response.returncode or not values.get("password"):
        raise RuntimeError("No cached GitHub credential was available; credential contents were not logged.")
    return values["password"]


def request(token, url, method="GET", body=None, content_type="application/json", size=None):
    if urllib.parse.urlparse(url).hostname not in {"api.github.com", "uploads.github.com"}:
        raise ValueError("Unexpected GitHub API host")
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10", "User-Agent": "DecisionICL-evidence-release",
        "Content-Type": content_type}
    if size is not None:
        headers["Content-Length"] = str(size)
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"GitHub API returned HTTP {error.code} for {method} {urllib.parse.urlparse(url).path}") from None


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--tag", default="decisionicl-evidence-20261009")
    parser.add_argument("--target", default="main")
    args = parser.parse_args()
    token = credentials()
    repo = request(token, API)
    print(json.dumps({"repository": repo["full_name"], "private": repo["private"],
        "push_permission": repo.get("permissions", {}).get("push")}), flush=True)
    if args.check:
        return
    if not args.directory:
        parser.error("--directory is required when publishing")
    directory = args.directory.resolve()
    release_file = directory / "release_assets.json"
    bundle = json.loads(release_file.read_text())
    for asset in bundle["assets"]:
        path = directory / asset["name"]
        if not path.is_file() or path.stat().st_size != asset["size"] or digest(path) != asset["sha256"]:
            raise ValueError("Prepared release asset failed verification: " + asset["name"])
    try:
        release = request(token, API + "/releases/tags/" + args.tag)
    except RuntimeError as error:
        if "HTTP 404" not in str(error):
            raise
        release = request(token, API + "/releases", "POST", {
            "tag_name": args.tag, "target_commitish": args.target,
            "name": "DecisionICL audited evidence bundle (2026-10-09)",
            "body": bundle["release_body"], "draft": True, "prerelease": True,
        })
    existing = {asset["name"]: asset for asset in request(token, release["assets_url"] + "?per_page=100")}
    receipts = []
    for asset in bundle["assets"]:
        if asset["name"] in existing:
            uploaded = existing[asset["name"]]
            if uploaded.get("digest") != "sha256:" + asset["sha256"] or uploaded["size"] != asset["size"]:
                raise ValueError("An existing release asset differs: " + asset["name"])
        else:
            url = release["upload_url"].split("{", 1)[0] + "?name=" + urllib.parse.quote(asset["name"])
            print(json.dumps({"stage": "upload", "name": asset["name"], "size": asset["size"]}), flush=True)
            with (directory / asset["name"]).open("rb") as stream:
                uploaded = request(token, url, "POST", stream,
                    content_type="application/octet-stream", size=asset["size"])
            if uploaded.get("digest") != "sha256:" + asset["sha256"] or uploaded["size"] != asset["size"]:
                raise ValueError("Uploaded asset SHA/size mismatch: " + asset["name"])
        receipts.append({key: uploaded[key] for key in ("name", "size", "digest", "browser_download_url")})
        print(json.dumps({"stage": "verified_upload", "name": asset["name"]}), flush=True)
    release = request(token, release["url"], "PATCH", {"draft": False, "body": bundle["release_body"]})
    # Publishing replaces draft "untagged-*" download URLs with the final tag.
    published_assets = {asset["name"]: asset for asset in request(token, release["assets_url"] + "?per_page=100")}
    receipts = []
    for expected in bundle["assets"]:
        asset = published_assets[expected["name"]]
        if asset.get("digest") != "sha256:" + expected["sha256"] or asset["size"] != expected["size"]:
            raise ValueError("Published release asset differs: " + expected["name"])
        receipts.append({key: asset[key] for key in ("name", "size", "digest", "browser_download_url")})
    result = {"release_url": release["html_url"], "tag": args.tag, "assets": receipts}
    (directory / "upload_receipt.json").write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
