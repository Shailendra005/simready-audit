import math

import numpy as np
import pytest

from helpers import box, cup_boxes, cylinder, merge
from simready_audit.analytics.mesh import mass_properties, shell_mass_properties
from simready_audit.io.types import TriMesh
from simready_audit.sim.cavity import cavity_probe
from simready_audit.sim.tests import critical_slide_angle, drop_test, drop_verdict, roll_test
from simready_audit.synth import write_asset


def test_box_drop_is_stable(tmp_path, cfg):
    v, f = box((0.05, 0.05, 0.05), (0, 0, 0.025))
    a = write_asset(tmp_path, "cube", "widget", {"vis": (v, f)}, (v, f), density=700.0)
    runs = drop_test(a["mjcf"], cfg, np.random.default_rng(0))
    problems, warns = drop_verdict(runs, cfg, with_warnings=True)
    assert not problems and not warns
    assert max(r["penetration_mm"] for r in runs.values()) < 0.05


@pytest.mark.parametrize("mu", [0.2, 0.5, 0.8])
def test_slide_angle_matches_coulomb(tmp_path, cfg, mu):
    v, f = box((0.05, 0.05, 0.05), (0, 0, 0.025))
    a = write_asset(tmp_path, f"block_{int(mu * 10)}", "widget", {"vis": (v, f)}, (v, f), density=700.0, friction=mu)
    theta = critical_slide_angle(a["mjcf"], cfg)
    assert theta == pytest.approx(math.degrees(math.atan(mu)), abs=0.3)


def test_roll_separates_solid_and_hollow_inertia(tmp_path, cfg):
    r, h = 0.033, 0.12
    v, f = cylinder(r, h, sections=1024)
    solid = mass_properties(v, f, 1.0)
    shell = shell_mass_properties(v, f)
    m = 0.01
    a = write_asset(tmp_path, "solid", "beverage_can", {"vis": (v, f)}, (v, f), mass=m, com=solid["com"], inertia=solid["inertia"] / solid["mass"] * m)
    b = write_asset(tmp_path, "hollow", "beverage_can", {"vis": (v, f)}, (v, f), mass=m, com=shell["com"], inertia=shell["inertia"] / shell["mass"] * m)
    rs, rh = roll_test(a["mjcf"], cfg), roll_test(b["mjcf"], cfg)
    assert rs["expected_from_inertia"] == pytest.approx(2 / 3, rel=0.01)
    assert rs["accel_ratio"] == pytest.approx(rs["expected_from_inertia"], rel=0.02)
    assert rh["accel_ratio"] == pytest.approx(rh["expected_from_inertia"], rel=0.02)
    assert (rs["accel_ratio"] / rh["accel_ratio"]) == pytest.approx(rs["expected_from_inertia"] / rh["expected_from_inertia"], rel=0.01)


def test_cavity_probe_enters_cup_and_is_blocked_by_hull(tmp_path, cfg):
    parts = cup_boxes()
    v, f = merge(parts)
    zone = TriMesh("vis_base__body", v, f)
    kw = dict(density=800.0, mass=0.3, com=np.array([0, 0, 0.04]), inertia=np.diag([3e-4, 3e-4, 3e-4]))
    hull = write_asset(tmp_path, "cup_hull", "mug", {"vis_base__body": (v, f)}, (v, f), **kw)
    split = write_asset(tmp_path, "cup_parts", "mug", {"vis_base__body": (v, f)}, (v, f), mjcf_hulls=parts, **kw)
    assert cavity_probe(hull["mjcf"], zone, cfg)["result"] == "BLOCKED"
    r = cavity_probe(split["mjcf"], zone, cfg)
    assert r["result"] == "ENTERED" and r["reached_floor"]
