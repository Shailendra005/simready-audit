import hashlib
import json
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "datasets.yaml"
DATA = ROOT / "data"


def load_datasets():
    return yaml.safe_load(CONFIG.read_text())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_dir(name):
    return DATA / name


def download(name):
    spec = load_datasets()[name]
    target = dataset_dir(name)
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="dataset",
        revision=spec["revision"],
        allow_patterns=spec["allow_patterns"],
        local_dir=target,
        max_workers=8,
    )
    files = sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts)
    lock = {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "files": {str(p.relative_to(target)): sha256_file(p) for p in files},
    }
    lock_path = ROOT / "configs" / f"{name}.lock.json"
    lock_path.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    return target, lock


def verify_lock(name):
    lock = json.loads((ROOT / "configs" / f"{name}.lock.json").read_text())
    target = dataset_dir(name)
    bad = [rel for rel, digest in lock["files"].items() if not (target / rel).exists() or sha256_file(target / rel) != digest]
    return lock, bad
