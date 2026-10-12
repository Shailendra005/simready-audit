import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .analytics import mesh as M
from .analytics.derived import AssetContext
from .audit import ROOT, load_config, seed_for
from .checks.base import run_checks
from .io.usd_loader import load_usd

ENGINE_DEFAULT_DENSITY = 1000.0
GENERIC = {"A1", "A2", "B1", "B2", "B4", "C2", "C4", "C5", "C6"}


def _objects(arm):
    root = ROOT / "data" / arm / "assets"
    return sorted(p for p in root.glob("*/*.usda"))


def _effective_mass(ctx, decomposition_threshold=0.05):
    u = ctx.usd
    if u.mass:
        return float(u.mass), "authored mass"
    dens = [c.material.density for c in u.colliders if c.material is not None and c.material.density]
    rho = dens[0] if dens else ENGINE_DEFAULT_DENSITY
    approx = {c.approximation for c in u.colliders}
    if "convexDecomposition" in approx:
        from .fixes.decompose import decompose
        hulls = decompose(ctx.collider_union or ctx.collider, threshold=decomposition_threshold, seed=0)
        parts = [M.convex_hull(M.TriMesh("h", v, f)) for v, f in hulls]
        un, _ = M.union(parts)
        vol = M.mass_properties(un.vertices, un.faces)["volume"]
        basis = "engine default density x approximate convex decomposition volume"
    else:
        vol = ctx.envelope_volume
        basis = "engine default density x convex hull volume"
    return float(rho * vol), basis


def _job(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    arm, path, cfg_path = args
    cfg, _ = load_config(cfg_path)
    u = load_usd(path)
    ctx = AssetContext(usd=u, mjcf=None, cfg=cfg, seed=seed_for(u.vid))
    checks = {r.check_id: r.to_dict() for r in run_checks(ctx) if r.check_id in GENERIC}
    lo, hi = ctx.collider.vertices.min(0), ctx.collider.vertices.max(0)
    vlo = np.min([v.vertices.min(0) for v in u.visuals] or [lo], axis=0)
    vhi = np.max([v.vertices.max(0) for v in u.visuals] or [hi], axis=0)
    mass, basis = _effective_mass(ctx)
    mats = [c.material for c in u.colliders if c.material is not None]
    return {
        "arm": arm,
        "object": path.parent.name,
        "checks": checks,
        "extent_cm": ((vhi - vlo) * 100).tolist(),
        "effective_mass_kg": mass,
        "mass_basis": basis,
        "friction": [mats[0].static_friction, mats[0].dynamic_friction] if mats else None,
        "approximations": sorted({str(c.approximation) for c in u.colliders}),
        "colliders": len(u.colliders),
    }


def run_controls(cfg_path=None, workers=8):
    jobs = [(arm, p, cfg_path) for arm in ("yam-authored", "yam-default") for p in _objects(arm)]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_job, jobs))
    by = {}
    for r in rows:
        by.setdefault(r["object"], {})[r["arm"]] = r
    table = []
    for obj, arms in sorted(by.items()):
        a, d = arms.get("yam-authored"), arms.get("yam-default")
        if not a or not d:
            continue
        ea, ed = np.sort(a["extent_cm"]), np.sort(d["extent_cm"])
        size_err = float(np.max(np.abs(ed - ea) / ea))
        bad = lambda r: sorted(k for k, c in r["checks"].items() if c["status"] in ("FAIL", "WARN"))
        table.append({
            "object": obj,
            "authored_flags": bad(a),
            "default_flags": bad(d),
            "authored_mass_kg": a["effective_mass_kg"],
            "default_mass_kg": d["effective_mass_kg"],
            "default_mass_ratio": d["effective_mass_kg"] / a["effective_mass_kg"],
            "default_size_err": size_err,
            "authored_extent_cm": a["extent_cm"],
            "default_extent_cm": d["extent_cm"],
            "default_friction": d["friction"],
            "authored_friction": a["friction"],
        })
    return {"rows": rows, "table": table}
