import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .audit import ROOT, audit_static, build_context, load_config
from .checks.base import run_checks
from .io.catalog import iter_variants

FIXED_DIR = ROOT / "results" / "fixed_mjcf"


def _static_job(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    v, cfg_path = args
    cfg, _ = load_config(cfg_path)
    _, res = audit_static(v["usda"], v["mjcf"], cfg)
    return {"vid": v["vid"], "cls": v["cls"], "checks": res}


def _full_job(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .sim.checks import sim_checks
    v, cfg_path = args
    cfg, _ = load_config(cfg_path)
    ctx, res = audit_static(v["usda"], v["mjcf"], cfg)
    if v["mjcf"] is not None:
        res += sim_checks(ctx, v["mjcf"])
    return {"vid": v["vid"], "cls": v["cls"], "checks": res}


def needs_fix(ctx):
    return ctx.usd.cls == "mug"


def _fix_job(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .fixes.decompose import decompose
    from .fixes.mjcf_rewrite import rewrite
    from .sim.checks import sim_checks
    v, cfg_path, threshold = args
    cfg, _ = load_config(cfg_path)
    ctx = build_context(v["usda"], v["mjcf"], cfg, compile_model=False)
    hulls = decompose(ctx.collider_union, threshold=threshold, seed=0)
    out_xml = FIXED_DIR / f"t{threshold}" / v["vid"] / f"{v['vid']}.xml"
    rewrite(v["mjcf"], out_xml, hulls=hulls, friction=ctx.material.static_friction)
    fctx = build_context(v["usda"], out_xml, cfg, compile_model=True)
    res = [r.to_dict() for r in run_checks(fctx, groups=("contract", "geometry"))]
    res = [r for r in res if r["check_id"] in ("A3", "A4", "A5", "A6", "A7", "B5")]
    res += sim_checks(fctx, out_xml, include=("D1", "D3", "D5"))
    return {"vid": v["vid"], "cls": v["cls"], "threshold": threshold, "hulls": len(hulls), "fixed_mjcf": str(out_xml.relative_to(ROOT)), "checks": res}


def _can_inertia_job(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import xml.etree.ElementTree as ET

    from .fixes.mjcf_rewrite import _abs_assets
    from .sim.tests import roll_test
    v, cfg_path = args
    cfg, _ = load_config(cfg_path)
    ctx = build_context(v["usda"], v["mjcf"], cfg, compile_model=False)
    k2 = ctx.shell["inertia"] / ctx.shell["mass"]
    inertia = k2 * ctx.usd.mass
    tree = ET.parse(v["mjcf"])
    root = tree.getroot()
    _abs_assets(root, Path(v["mjcf"]).parent)
    i = root.find("worldbody").find("body").find("inertial")
    i.set("fullinertia", " ".join(f"{x:.9g}" for x in (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])))
    out = FIXED_DIR / "shell_inertia" / v["vid"] / f"{v['vid']}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    before = roll_test(v["mjcf"], cfg)
    after = roll_test(out, cfg)
    return {"vid": v["vid"], "before": before, "after": after}


def run_can_inertia(dataset_root, cfg_path=None, workers=8):
    vs = [v for v in select(dataset_root) if v["cls"] == "beverage_can"]
    return _map(_can_inertia_job, [(v, cfg_path) for v in vs], workers)


def _map(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs, chunksize=1))


def select(dataset_root, limit=None, vids=None, cls=None):
    vs = list(iter_variants(dataset_root))
    if vids:
        vs = [v for v in vs if v["vid"] in set(vids)]
    if cls:
        vs = [v for v in vs if v["cls"] == cls]
    return vs[:limit] if limit else vs


def run_static(dataset_root, cfg_path=None, workers=8, limit=None):
    return _map(_static_job, [(v, cfg_path) for v in select(dataset_root, limit)], workers)


def run_full(dataset_root, cfg_path=None, workers=8, limit=None, vids=None):
    return _map(_full_job, [(v, cfg_path) for v in select(dataset_root, limit, vids)], workers)


def run_fix(dataset_root, cfg_path=None, workers=8, threshold=0.05, limit=None, vids=None):
    vs = [v for v in select(dataset_root, limit, vids) if v["cls"] == "mug"]
    return _map(_fix_job, [(v, cfg_path, threshold) for v in vs], workers)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
