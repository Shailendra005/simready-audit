import hashlib
from pathlib import Path

import yaml

from .analytics.derived import AssetContext
from .checks import contract, geometry, physical
from .checks.base import run_checks
from .io.mjcf_loader import load_mjcf
from .io.usd_loader import load_usd

CHECK_MODULES = (contract, geometry, physical)

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path or ROOT / "configs" / "thresholds.v1.yaml")
    raw = path.read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def seed_for(vid, salt="simready"):
    return int(hashlib.sha256(f"{salt}:{vid}".encode()).hexdigest()[:8], 16)


def build_context(usda, mjcf, cfg, compile_model=True):
    usd = load_usd(usda)
    mj = load_mjcf(mjcf, compile_model=compile_model) if mjcf else None
    return AssetContext(usd=usd, mjcf=mj, cfg=cfg, seed=seed_for(usd.vid))


def audit_static(usda, mjcf, cfg):
    ctx = build_context(usda, mjcf, cfg)
    return ctx, [r.to_dict() for r in run_checks(ctx)]
