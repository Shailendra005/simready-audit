import argparse
import csv
import json
import sys
import time

from .audit import ROOT, load_config

RESULTS = ROOT / "results"


def _dataset_root(name):
    from .data.download import dataset_dir
    return dataset_dir(name)


def cmd_download(a):
    from .data.download import download
    target, lock = download(a.dataset)
    print(f"{lock['repo_id']}@{lock['revision'][:12]}: {len(lock['files'])} files in {target}")


def _require_frozen(a):
    from .provenance import frozen_hash
    _, h = load_config(a.config)
    fh = frozen_hash()
    if fh and h != fh and not a.allow_unfrozen:
        sys.exit(f"config hash {h[:12]} does not match FROZEN.sha256 {fh[:12]}; pass --allow-unfrozen for exploratory runs")
    return h


def flatten(rows):
    out = []
    for r in rows:
        row = {"vid": r["vid"], "cls": r["cls"]}
        for c in r["checks"]:
            row[c["check_id"]] = c["status"]
        out.append(row)
    return out


def write_csv(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def cmd_audit(a):
    from .provenance import canonical_hash, stamp
    from .runner import run_full, run_static, write_json
    _require_frozen(a)
    t = time.time()
    root = _dataset_root(a.dataset)
    rows = (run_static if a.static_only else run_full)(root, a.config, workers=a.workers, limit=a.limit)
    rows = sorted(rows, key=lambda r: r["vid"])
    out = RESULTS / a.dataset
    for r in rows:
        write_json(r, out / "assets" / f"{r['vid']}.json")
    write_csv(flatten(rows), out / "results.csv")
    meta = stamp(a.dataset, a.config)
    meta.update({"assets": len(rows), "static_only": a.static_only, "results_sha256": canonical_hash(rows), "elapsed_s": round(time.time() - t, 1)})
    write_json(meta, out / "run.json")
    print(json.dumps({k: meta[k] for k in ("assets", "results_sha256", "config_sha256", "config_frozen", "elapsed_s")}, indent=1))


def cmd_fix(a):
    from .runner import run_fix, write_json
    _require_frozen(a)
    if not a.cans_only:
        rows = sorted(run_fix(_dataset_root(a.dataset), a.config, workers=a.workers, threshold=a.threshold, limit=a.limit), key=lambda r: r["vid"])
        write_json(rows, RESULTS / a.dataset / f"fixed_t{a.threshold}.json")
        print(f"fixed {len(rows)} mugs")
    from .runner import run_can_inertia
    cans = sorted(run_can_inertia(_dataset_root(a.dataset), a.config, workers=a.workers), key=lambda r: r["vid"])
    write_json(cans, RESULTS / a.dataset / "can_shell_inertia.json")
    print(f"can inertia demo on {len(cans)} cans")


def cmd_mutate(a):
    from .mutate.study import run_study
    from .runner import write_json
    _require_frozen(a)
    s = run_study(_dataset_root(a.dataset), a.config, workers=a.workers, per_class=a.per_class)
    write_json(s, RESULTS / a.dataset / "mutation.json")
    for d in s["detection"]:
        print(f"{d['mutation']:28s} {d['detected']}/{d['eligible']}")


def cmd_controls(a):
    from .controls import run_controls
    from .provenance import stamp
    from .runner import write_json
    _require_frozen(a)
    s = run_controls(a.config, workers=a.workers)
    s["stamp"] = {k: stamp(k, a.config)["dataset_revision"] for k in ("yam-authored", "yam-default")}
    write_json(s, RESULTS / "paper-controls.json")
    flagged = lambda arm: sum(1 for r in s["table"] if r[f"{arm}_flags"])
    print(f"authored arm flagged {flagged('authored')}/{len(s['table'])}, default arm flagged {flagged('default')}/{len(s['table'])}")


def cmd_report(a):
    from .report.build import build_all
    for p in build_all(a.dataset):
        print(p)


def cmd_freeze(a):
    _, h = load_config(a.config)
    (ROOT / "configs" / "FROZEN.sha256").write_text(f"{h}  configs/thresholds.v1.yaml\n")
    print(h)


def cmd_verify(a):
    from .data.download import verify_lock
    from .provenance import canonical_hash, frozen_hash
    from .runner import run_full, run_static
    ok = True
    lock, bad = verify_lock(a.dataset)
    print(f"dataset files: {len(lock['files'])}, mismatched: {len(bad)}")
    ok &= not bad
    _, h = load_config(a.config)
    print(f"config frozen: {h == frozen_hash()}")
    ok &= h == frozen_hash()
    root = _dataset_root(a.dataset)
    fn = run_static if a.static_only else run_full
    r1 = sorted(fn(root, a.config, workers=1, limit=a.limit), key=lambda r: r["vid"])
    r2 = sorted(fn(root, a.config, workers=a.workers, limit=a.limit), key=lambda r: r["vid"])
    h1, h2 = canonical_hash(r1), canonical_hash(r2)
    print(f"determinism (workers=1 vs {a.workers}, {len(r1)} assets): {h1[:16]} vs {h2[:16]} -> {'identical' if h1 == h2 else 'DIFFERENT'}")
    ok &= h1 == h2
    sys.exit(0 if ok else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="simready-audit")
    p.add_argument("--dataset", default="bottles-and-containers")
    p.add_argument("--config", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--allow-unfrozen", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(fn=cmd_download)
    s = sub.add_parser("audit")
    s.add_argument("--static-only", action="store_true")
    s.add_argument("--limit", type=int)
    s.set_defaults(fn=cmd_audit)
    s = sub.add_parser("fix")
    s.add_argument("--threshold", type=float, default=0.05)
    s.add_argument("--cans-only", action="store_true")
    s.add_argument("--limit", type=int)
    s.set_defaults(fn=cmd_fix)
    s = sub.add_parser("mutate")
    s.add_argument("--per-class", type=int, default=10)
    s.set_defaults(fn=cmd_mutate)
    sub.add_parser("controls").set_defaults(fn=cmd_controls)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    sub.add_parser("freeze").set_defaults(fn=cmd_freeze)
    s = sub.add_parser("verify")
    s.add_argument("--limit", type=int, default=12)
    s.add_argument("--static-only", action="store_true")
    s.set_defaults(fn=cmd_verify)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
