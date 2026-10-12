import json
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..audit import ROOT, load_config
from .charts import detection_chart
from .figures import S2, S3, _rgba, probe_pair_figure, probe_scene, render
from .stats import compute

TEMPLATES = Path(__file__).parent / "templates"
PUBLIC = ROOT / "reports" / "public"
PRIVATE_SRC = ROOT / "private"

RULES = {
    "A1": "stage units are not metres / kilograms / Z-up",
    "A2": "no rigid body, no mass and no density, no collider, unbound physics material, invalid friction or restitution",
    "A3": "the MJCF does not compile",
    "A4": "USD and MJCF mass differ by more than 1e-5 (relative)",
    "A5": "USD and MJCF centre of mass differ by more than 1 µm",
    "A6": "USD and MJCF inertia tensors differ by more than 1e-4 (relative Frobenius)",
    "A7": "MJCF sliding friction lies outside the USD [dynamic, static] interval",
    "A8": "USD restitution > 0.05 but the MJCF contact is critically damped (warn)",
    "A9": "information: USD collider approximation vs number of MJCF mesh geoms",
    "A10": "max_mass_kg equals the physics mass, exceeds the convex envelope, or is >25% off the expected capacity",
    "B1": "collider has boundary or non-manifold edges",
    "B2": "inconsistent or inward winding",
    "B3": "faces smaller than 1e-12 m² (warn)",
    "B4": "information: connected components",
    "B5": "over 2% of the MuJoCo collision surface lies more than 2 mm from the authored collider (warn)",
    "B6": "MJCF collision vertices sit more than 1 mm off the USD collider (warn)",
    "C1": "dimensions outside the dataset card's stated formats (warn)",
    "C2": "mass differs from density × volume by >5% (warn) or >20% (fail); thin-walled classes reported only",
    "C3": "implied wall thinner than the thinnest commercial can wall, 0.097 mm (warn)",
    "C4": "inertia not positive definite or violates the triangle inequality",
    "C5": "inertia per unit mass outside the band any solid-to-shell mass distribution in this shape allows (thin-walled classes must be shell-like)",
    "C6": "COM outside the bounding box (fail) or >2% of the diagonal away from the solid/shell centroids (warn)",
    "D1": "drop from 14 orientations: NaN, energy gain >1%, penetration >0.5 mm (fail); upright drop that never settles (warn)",
    "D2": "instability at 0.5 / 1 / 2 ms (fail); penetration or upright outcome changes (warn)",
    "D3": "critical slide angle outside atan(μd)…atan(μs) of the USD material by more than 2°",
    "D4": "information: rolling acceleration of cans vs the value their inertia implies",
    "D5": "a ball dropped into an open container stays on top of it",
}


def fmt_cell(c):
    if c["n"] == 0 or c["pass"] + c["warn"] + c["fail"] == 0:
        return "–"
    return f"{c['pass']} / {c['warn']} / {c['fail']}"


def count_tests():
    out = subprocess.run(["uv", "run", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if "::" in line)


def env():
    e = Environment(loader=FileSystemLoader(str(TEMPLATES)), undefined=StrictUndefined, trim_blocks=False, lstrip_blocks=False)
    return e


def fixture_figure(cfg, out_png):
    import tempfile

    import numpy as np
    import trimesh

    from ..io.types import TriMesh
    from ..synth import write_asset

    def box(ext, c):
        m = trimesh.creation.box(extents=ext)
        m.apply_translation(c)
        return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)

    outer, height, wall = 0.08, 0.09, 0.006
    half = outer / 2 - wall / 2
    parts = [box((outer, outer, wall), (0, 0, wall / 2))] + [box((ex, ey, height), (dx, dy, height / 2)) for dx, dy, ex, ey in ((half, 0, wall, outer), (-half, 0, wall, outer), (0, half, outer, wall), (0, -half, outer, wall))]
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    v, f = np.vstack(vs), np.vstack(fs)
    kw = dict(density=800.0, mass=0.3, com=np.array([0, 0, 0.04]), inertia=np.diag([3e-4, 3e-4, 3e-4]))
    with tempfile.TemporaryDirectory() as tmp:
        hull = write_asset(tmp, "cup_hull", "mug", {"vis_base__body": (v, f)}, (v, f), **kw)
        split = write_asset(tmp, "cup_parts", "mug", {"vis_base__body": (v, f)}, (v, f), mjcf_hulls=parts, **kw)
        zone = TriMesh("vis_base__body", v, f)
        m1, d1, _ = probe_scene(hull["mjcf"], zone, cfg, _rgba(S2, 0.30))
        m2, d2, _ = probe_scene(split["mjcf"], zone, cfg, _rgba(S3, 0.30))
        a = render(m1, d1, [0, 0, 0.045], 0.26, elevation=-32)
        b = render(m2, d2, [0, 0, 0.045], 0.26, elevation=-32)
    return probe_pair_figure((a, "One convex hull (orange): the ball rests on the rim."), (b, "Five convex boxes (green): the ball reaches the floor."), out_png, "Cup as one hull", "Cup as five boxes")


def build_all(dataset="bottles-and-containers"):
    cfg, _ = load_config()
    s = compute(dataset)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    tests = count_tests()
    s["tests"] = tests
    outputs = []

    fixture_figure(cfg, PUBLIC / "fixture_probe.png")
    if s.get("mutation"):
        detection_chart(s["mutation"]["sweeps"], PUBLIC / "detection_sweeps.png")
    checks = [{"id": r["id"], "title": r["title"], "rule": RULES.get(r["id"], "")} for r in s["table"]]
    readme = env().get_template("README.md.j2").render(checks=checks, fmt=fmt_cell, **s)
    (ROOT / "README.md").write_text(readme)
    outputs.append(ROOT / "README.md")
    (PUBLIC / "summary.json").write_text(json.dumps({k: s[k] for k in ("n_total", "n", "table", "tests")} | {"run": s["run"], "mutation": {k: s["mutation"][k] for k in ("assets", "types", "perfect", "inv_changes", "inv_comparisons", "rows", "sweeps")}}, indent=1, default=str))

    hook = PRIVATE_SRC / "private_report.py"
    if hook.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("private_report", hook)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        outputs += mod.build(dataset, cfg, tests)
    return outputs
