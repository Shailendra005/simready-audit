import math

import numpy as np
import pytest
import trimesh

from simready_audit.analytics import mesh as M
from simready_audit.io.types import TriMesh

from helpers import box, cylinder


def polygon_area(r, n):
    return 0.5 * n * r * r * math.sin(2 * math.pi / n)


def polygon_polar(r, n):
    a = 2 * math.pi / n
    return n * r ** 4 / 12 * math.sin(a) * (2 + math.cos(a))


def test_box_volume_and_inertia_exact():
    v, f = box((0.1, 0.2, 0.3))
    p = M.mass_properties(v, f, density=1000.0)
    assert p["volume"] == pytest.approx(0.006, rel=1e-12)
    expected = 6.0 / 12 * np.array([0.2 ** 2 + 0.3 ** 2, 0.1 ** 2 + 0.3 ** 2, 0.1 ** 2 + 0.2 ** 2])
    assert np.allclose(np.diag(p["inertia"]), expected, rtol=1e-12)
    assert np.allclose(p["com"], 0, atol=1e-15)


@pytest.mark.parametrize("n", [8, 32, 256])
def test_prism_matches_polygon_formulas(n):
    r, h, rho = 0.033, 0.115, 2700.0
    v, f = cylinder(r, h, sections=n)
    p = M.mass_properties(v, f, rho)
    assert p["volume"] == pytest.approx(polygon_area(r, n) * h, rel=1e-10)
    assert p["inertia"][2, 2] == pytest.approx(rho * h * polygon_polar(r, n), rel=1e-10)
    assert p["com"][2] == pytest.approx(h / 2, rel=1e-12)


def test_thick_cup_volume_exact():
    n, ro, ri, h, t = 128, 0.04, 0.035, 0.09, 0.006
    outer = M.to_manifold(TriMesh("o", *cylinder(ro, h, n)))
    inner = M.to_manifold(TriMesh("i", *cylinder(ri, h, n, z0=t)))
    cup = M.manifold_to_trimesh(outer - inner, "cup")
    p = M.mass_properties(cup.vertices, cup.faces)
    expected = polygon_area(ro, n) * h - polygon_area(ri, n) * (h - t)
    assert p["volume"] == pytest.approx(expected, rel=1e-5)


def test_agrees_with_trimesh_on_irregular_mesh():
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    m.apply_transform(np.diag([1.0, 1.7, 0.6, 1.0]))
    m.apply_translation([0.01, -0.02, 0.03])
    p = M.mass_properties(np.asarray(m.vertices), np.asarray(m.faces), 1.0)
    assert p["volume"] == pytest.approx(m.volume, rel=1e-10)
    assert np.allclose(p["com"], m.center_mass, atol=1e-12)
    assert np.allclose(p["inertia"], m.moment_inertia, rtol=1e-9, atol=1e-15)


def test_thin_shell_cylinder_inertia():
    r, h = 0.033, 0.115
    v, f = cylinder(r, h, sections=1024)
    s = M.shell_mass_properties(v, f)
    side, cap = 2 * math.pi * r * h, math.pi * r * r
    expected = (side * r * r + 2 * cap * r * r / 2) / (side + 2 * cap)
    assert s["inertia"][2, 2] / s["mass"] == pytest.approx(expected, rel=1e-4)


def test_topology_detects_holes_and_flips():
    v, f = box((1, 1, 1))
    assert M.edge_topology(f)["boundary_edges"] == 0
    assert M.edge_topology(f[1:])["boundary_edges"] == 3
    flipped = f.copy()
    flipped[0] = flipped[0][::-1]
    assert M.edge_topology(flipped)["inconsistent_winding_edges"] > 0
    assert M.mass_properties(v, f[:, ::-1])["volume"] < 0


def test_union_removes_overlap():
    a = TriMesh("a", *box((0.1, 0.1, 0.1)))
    b = TriMesh("b", *box((0.1, 0.1, 0.1), (0.05, 0, 0)))
    u, bad = M.union([a, b])
    assert not bad
    assert M.mass_properties(u.vertices, u.faces)["volume"] == pytest.approx(0.0015, rel=1e-5)


def test_convex_hull_of_l_shape():
    a = box((0.2, 0.1, 0.1), (0.1, 0.05, 0.05))
    b = box((0.1, 0.2, 0.1), (0.05, 0.1, 0.05))
    v = np.vstack([a[0], b[0]])
    h = M.convex_hull(TriMesh("l", v, np.zeros((0, 3), int)))
    p = M.mass_properties(h.vertices, h.faces)
    assert p["volume"] == pytest.approx(0.2 * 0.2 * 0.1 - 0.5 * 0.1 * 0.1 * 0.1, rel=1e-9)
    assert M.edge_topology(h.faces)["inconsistent_winding_edges"] == 0


def test_uv_seam_duplicates_are_not_holes():
    v, f = box((1, 1, 1))
    dup = np.vstack([v, v[:2]])
    f2 = f.copy()
    f2[f2 == 0] = len(v)
    f2[f2 == 1] = len(v) + 1
    f2[:4] = f[:4]
    assert M.edge_topology(f2)["boundary_edges"] > 0
    t, _, welded = M.closed_topology(dup, f2)
    assert welded and t["boundary_edges"] == 0
    t2, _, _ = M.closed_topology(v, f[1:])
    assert t2["boundary_edges"] == 3
