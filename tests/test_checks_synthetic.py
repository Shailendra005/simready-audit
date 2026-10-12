import xml.etree.ElementTree as ET

import numpy as np

from helpers import cylinder
from simready_audit.audit import build_context
from simready_audit.checks.base import run_checks
from simready_audit.synth import write_asset

BAD = {"FAIL", "WARN"}


def statuses(a, cfg):
    ctx = build_context(a["usda"], a["mjcf"], cfg)
    return {r.check_id: r.status for r in run_checks(ctx)}


def clean(tmp_path, name="clean", **kw):
    v, f = cylinder(0.04, 0.1, sections=64)
    kw.setdefault("density", 800.0)
    kw.setdefault("sf", 0.5)
    kw.setdefault("df", 0.4)
    kw.setdefault("friction", 0.45)
    kw.setdefault("e", 0.0)
    return write_asset(tmp_path, name, "widget", {"vis": (v, f)}, (v, f), **kw)


def test_clean_asset_raises_nothing(tmp_path, cfg):
    s = statuses(clean(tmp_path), cfg)
    assert not {k: v for k, v in s.items() if v in BAD and k != "A10"}, s


def test_units(tmp_path, cfg):
    assert statuses(clean(tmp_path, meters_per_unit=0.01), cfg)["A1"] == "FAIL"


def test_open_and_flipped_collider(tmp_path, cfg):
    v, f = cylinder(0.04, 0.1, sections=64)
    a = write_asset(tmp_path, "open", "widget", {"vis": (v, f)}, (v, f[5:]))
    assert statuses(a, cfg)["B1"] == "FAIL"
    b = write_asset(tmp_path, "flip", "widget", {"vis": (v, f)}, (v, f[:, ::-1]), mass=0.4, com=np.zeros(3), inertia=np.diag([1e-4, 1e-4, 1e-4]))
    assert statuses(b, cfg)["B2"] == "FAIL"


def _edit_mjcf(a, fn):
    tree = ET.parse(a["mjcf"])
    fn(tree.getroot())
    tree.write(a["mjcf"])
    return a


def test_parity_checks(tmp_path, cfg):
    a = _edit_mjcf(clean(tmp_path, "m"), lambda r: r.find(".//inertial").set("mass", str(float(r.find(".//inertial").get("mass")) * 1.01)))
    assert statuses(a, cfg)["A4"] == "FAIL"
    b = _edit_mjcf(clean(tmp_path, "c"), lambda r: r.find(".//inertial").set("pos", "0.001 0 0.05"))
    assert statuses(b, cfg)["A5"] == "FAIL"
    def swap(r):
        i = r.find(".//inertial")
        vals = i.get("fullinertia").split()
        vals[0], vals[2] = vals[2], vals[0]
        i.set("fullinertia", " ".join(vals))
    c = _edit_mjcf(clean(tmp_path, "i"), swap)
    assert statuses(c, cfg)["A6"] == "FAIL"


def test_friction_outside_interval(tmp_path, cfg):
    assert statuses(clean(tmp_path, friction=0.2), cfg)["A7"] == "FAIL"
    assert statuses(clean(tmp_path, "ok", friction=0.45), cfg)["A7"] == "PASS"


def test_max_mass_equal_to_mass(tmp_path, cfg):
    v, f = cylinder(0.04, 0.1, sections=64)
    a = write_asset(tmp_path, "mm", "widget", {"vis": (v, f)}, (v, f), density=800.0, max_mass=800.0 * np.pi * 0.04 ** 2 * 0.1)
    assert statuses(a, cfg)["A10"] == "FAIL"


def test_mass_inconsistent_with_volume(tmp_path, cfg):
    v, f = cylinder(0.04, 0.1, sections=64)
    from simready_audit.analytics.mesh import mass_properties
    p = mass_properties(v, f, 800.0)
    a = write_asset(tmp_path, "heavy", "widget", {"vis": (v, f)}, (v, f), density=800.0, mass=p["mass"] * 1.5, com=p["com"], inertia=p["inertia"] * 1.5)
    assert statuses(a, cfg)["C2"] == "FAIL"


def test_inertia_inconsistent_with_mass_distribution(tmp_path, cfg):
    v, f = cylinder(0.04, 0.1, sections=64)
    from simready_audit.analytics.mesh import mass_properties
    p = mass_properties(v, f, 800.0)
    i = p["inertia"].copy()
    i[[0, 2], [0, 2]] = i[[2, 0], [2, 0]]
    a = write_asset(tmp_path, "swap", "widget", {"vis": (v, f)}, (v, f), density=800.0, mass=p["mass"], com=p["com"], inertia=i)
    assert statuses(a, cfg)["C5"] == "FAIL"


def test_phantom_surface_from_single_hull(tmp_path, cfg):
    from helpers import cup_boxes, merge
    parts = cup_boxes()
    v, f = merge(parts)
    a = write_asset(tmp_path, "cup_hull", "widget", {"vis": (v, f)}, (v, f), density=800.0, mass=0.3, com=np.array([0, 0, 0.04]), inertia=np.diag([3e-4, 3e-4, 3e-4]))
    b = write_asset(tmp_path, "cup_parts", "widget", {"vis": (v, f)}, (v, f), density=800.0, mass=0.3, com=np.array([0, 0, 0.04]), inertia=np.diag([3e-4, 3e-4, 3e-4]), mjcf_hulls=parts)
    assert statuses(a, cfg)["B5"] == "WARN"
    assert statuses(b, cfg)["B5"] == "PASS"
