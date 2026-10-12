import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from math import sqrt

import numpy as np

from ..audit import audit_static, load_config
from ..io.catalog import iter_variants
from .mutations import INVARIANCES, MUTATIONS, SWEEPS, materialize

BAD = {"FAIL", "WARN"}


def pick_assets(dataset_root, per_class=10, seed=7):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for v in iter_variants(dataset_root):
        by[v["cls"]].append(v)
    out = []
    for cls in sorted(by):
        vs = by[cls]
        idx = rng.choice(len(vs), size=min(per_class, len(vs)), replace=False)
        out.extend(vs[i] for i in sorted(idx))
    return out


def _statuses(usda, mjcf, cfg):
    _, res = audit_static(usda, mjcf, cfg)
    return {r["check_id"]: r["status"] for r in res}


def _job(args):
    v, cfg_path, kind, name, fn, arg = args
    cfg, _ = load_config(cfg_path)
    with tempfile.TemporaryDirectory() as tmp:
        if kind == "baseline":
            usda, mjcf = v["usda"], v["mjcf"]
        elif kind == "invariance":
            usda, mjcf = fn(v["usda"], v["mjcf"], tmp, arg) if arg is not None else fn(v["usda"], v["mjcf"], tmp)
        else:
            usda, mjcf = materialize(v["usda"], v["mjcf"], tmp, fn, arg)
        st = _statuses(usda, mjcf, cfg)
    return {"vid": v["vid"], "cls": v["cls"], "kind": kind, "name": name, "arg": arg if not callable(arg) else None, "statuses": st}


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def run_study(dataset_root, cfg_path=None, workers=8, per_class=10, progress=True):
    assets = pick_assets(dataset_root, per_class)
    jobs = [(v, cfg_path, "baseline", "baseline", None, None) for v in assets]
    for name, (fn, arg, _) in MUTATIONS.items():
        jobs += [(v, cfg_path, "mutation", name, fn, arg) for v in assets]
    for name, (fn, args, _) in SWEEPS.items():
        for a in args:
            jobs += [(v, cfg_path, "sweep", f"{name}@{a}", fn, a) for v in assets]
    for name, (fn, arg) in INVARIANCES.items():
        jobs += [(v, cfg_path, "invariance", name, fn, arg) for v in assets]
    rows = []
    from concurrent.futures import as_completed
    import sys
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if progress and (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)}", file=sys.stderr, flush=True)
    return summarize(rows, assets)


def summarize(rows, assets):
    base = {r["vid"]: r["statuses"] for r in rows if r["kind"] == "baseline"}
    detection = []
    for name, (_, arg, targets) in MUTATIONS.items():
        eligible = [r for r in rows if r["kind"] == "mutation" and r["name"] == name and all(base[r["vid"]].get(t) not in BAD for t in targets)]
        caught = [r for r in eligible if any(r["statuses"].get(t) in BAD for t in targets)]
        lo, hi = wilson(len(caught), len(eligible))
        detection.append({"mutation": name, "targets": targets, "eligible": len(eligible), "detected": len(caught), "rate": (len(caught) / len(eligible)) if eligible else None, "ci95": [lo, hi], "excluded_already_failing": len(assets) - len(eligible)})
    sweeps = []
    for name, (_, args, targets) in SWEEPS.items():
        for a in args:
            key = f"{name}@{a}"
            eligible = [r for r in rows if r["kind"] == "sweep" and r["name"] == key and all(base[r["vid"]].get(t) not in BAD for t in targets)]
            caught = [r for r in eligible if any(r["statuses"].get(t) in BAD for t in targets)]
            sweeps.append({"sweep": name, "magnitude": a, "targets": targets, "eligible": len(eligible), "detected": len(caught), "rate": (len(caught) / len(eligible)) if eligible else None})
    invariance = []
    for name in INVARIANCES:
        rs = [r for r in rows if r["kind"] == "invariance" and r["name"] == name]
        changed = []
        for r in rs:
            diff = {k: (base[r["vid"]].get(k), s) for k, s in r["statuses"].items() if base[r["vid"]].get(k) in ("PASS", "WARN", "FAIL") and base[r["vid"]].get(k) != s}
            if diff:
                changed.append({"vid": r["vid"], "diff": diff})
        comparisons = sum(1 for r in rs for k in r["statuses"] if base[r["vid"]].get(k) in ("PASS", "WARN", "FAIL"))
        flips = sum(len(c["diff"]) for c in changed)
        lo, hi = wilson(flips, comparisons)
        invariance.append({"transform": name, "assets": len(rs), "verdict_comparisons": comparisons, "verdict_changes": flips, "ci95": [lo, hi], "examples": changed[:5]})
    return {"assets": [a["vid"] for a in assets], "detection": detection, "sweeps": sweeps, "invariance": invariance}
