import hashlib
import json
import platform
import subprocess

from .audit import ROOT, load_config


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def versions():
    from importlib.metadata import version
    out = {"python": platform.python_version()}
    for pkg in ("mujoco", "usd-core", "trimesh", "numpy", "scipy", "manifold3d", "coacd"):
        out[pkg] = version(pkg)
    return out


def frozen_hash():
    p = ROOT / "configs" / "FROZEN.sha256"
    return p.read_text().split()[0] if p.exists() else None


def stamp(dataset, cfg_path=None):
    _, h = load_config(cfg_path)
    lock = json.loads((ROOT / "configs" / f"{dataset}.lock.json").read_text())
    lockfile = ROOT / "uv.lock"
    return {
        "config_sha256": h,
        "config_frozen": h == frozen_hash(),
        "dataset": lock["repo_id"],
        "dataset_revision": lock["revision"],
        "dataset_files": len(lock["files"]),
        "uv_lock_sha256": sha256_bytes(lockfile.read_bytes()) if lockfile.exists() else None,
        "git_commit": git_commit(),
        "versions": versions(),
    }


def canonical_hash(obj):
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())
