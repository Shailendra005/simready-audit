import json
from collections import Counter, defaultdict
from pathlib import Path


from ..audit import ROOT

CLASSES = ["beverage_can", "glass_jar", "mug"]
CLASS_LABEL = {"beverage_can": "Cans", "glass_jar": "Jars", "mug": "Mugs"}


def _load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def load_all(dataset="bottles-and-containers"):
    base = ROOT / "results" / dataset
    assets = [json.loads(p.read_text()) for p in sorted((base / "assets").glob("*.json"))]
    return {
        "run": _load(base / "run.json"),
        "assets": assets,
        "fixed": _load(base / "fixed_t0.05.json"),
        "cans": _load(base / "can_shell_inertia.json"),
        "mutation": _load(base / "mutation.json"),
        "controls": _load(ROOT / "results" / "paper-controls.json"),
    }


def check_index(assets):
    return {a["vid"]: {c["check_id"]: c for c in a["checks"]} for a in assets}


def status_table(assets):
    tab = defaultdict(lambda: defaultdict(Counter))
    titles = {}
    for a in assets:
        for c in a["checks"]:
            tab[c["check_id"]][a["cls"]][c["status"]] += 1
            titles[c["check_id"]] = c["title"]
    order = sorted(tab, key=lambda s: (s[0], int(s[1:])))
    rows = []
    for cid in order:
        row = {"id": cid, "title": titles[cid]}
        for cls in CLASSES:
            cnt = tab[cid][cls]
            n = sum(cnt.values())
            row[cls] = {"pass": cnt["PASS"], "warn": cnt["WARN"], "fail": cnt["FAIL"], "info": cnt["INFO"], "skip": cnt["SKIP"], "n": n}
        rows.append(row)
    return rows


def _metric(idx, cid, key, cls=None, assets=None):
    vals = []
    for a in assets:
        if cls and a["cls"] != cls:
            continue
        m = idx[a["vid"]].get(cid, {}).get("metrics", {})
        if key in m and m[key] is not None:
            vals.append(m[key])
    return vals


def compute(dataset="bottles-and-containers"):
    d = load_all(dataset)
    assets = d["assets"]
    s = {"run": d["run"], "n_total": len(assets), "n": dict(Counter(a["cls"] for a in assets)), "table": status_table(assets)}
    mut = d["mutation"]
    if mut:
        det = mut["detection"]
        s["mutation"] = {
            "assets": len(mut["assets"]),
            "types": len(det),
            "scored": [x for x in det if x["eligible"]],
            "perfect": sum(1 for x in det if x["eligible"] and x["detected"] == x["eligible"]),
            "rows": det,
            "sweeps": mut["sweeps"],
            "invariance": mut["invariance"],
            "inv_changes": sum(x["verdict_changes"] for x in mut["invariance"]),
            "inv_comparisons": sum(x["verdict_comparisons"] for x in mut["invariance"]),
        }
    return s
