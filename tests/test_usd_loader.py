import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

from helpers import box
from simready_audit.io.usd_loader import load_usd
from simready_audit.synth import write_asset


def test_units_and_up_axis_are_normalised(tmp_path):
    v, f = box((10.0, 20.0, 30.0), (0, 0, 15.0))
    a = write_asset(tmp_path, "cm_yup", "widget", {"vis": (v, f)}, (v, f), meters_per_unit=0.01, up_axis="Y")
    u = load_usd(a["usda"])
    assert u.meters_per_unit == 0.01 and u.up_axis == "Y"
    ext = u.colliders[0].mesh.vertices.max(0) - u.colliders[0].mesh.vertices.min(0)
    assert np.allclose(ext, [0.10, 0.30, 0.20], atol=1e-7)


def test_principal_axes_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    inertia = q @ np.diag([1e-4, 2e-4, 2.6e-4]) @ q.T
    v, f = box((0.1, 0.1, 0.1))
    a = write_asset(tmp_path, "rot", "widget", {"vis": (v, f)}, (v, f), mass=0.5, com=np.zeros(3), inertia=inertia)
    u = load_usd(a["usda"])
    assert np.allclose(u.inertia_tensor, inertia, rtol=1e-5, atol=1e-10)


def test_child_xform_stack_and_quads(tmp_path):
    path = tmp_path / "x.usda"
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/obj")
    st.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(5, 5, 5))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    child = UsdGeom.Mesh.Define(st, "/obj/quad")
    child.AddTranslateOp().Set(Gf.Vec3d(1, 0, 0))
    child.AddRotateZOp().Set(90.0)
    child.AddScaleOp().Set(Gf.Vec3f(2, 2, 2))
    child.CreatePointsAttr(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]))
    child.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    child.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    st.GetRootLayer().Save()
    u = load_usd(path)
    m = u.visuals[0]
    assert m.faces.shape == (2, 3)
    assert np.allclose(m.vertices[1], [1, 2, 0], atol=1e-6)
    assert np.allclose(m.vertices[2], [-1, 2, 0], atol=1e-6)


def test_real_asset_parity_with_mjcf():
    from pathlib import Path
    from simready_audit.io.mjcf_loader import load_mjcf
    root = Path(__file__).resolve().parents[1] / "data" / "bottles-and-containers" / "assets" / "mug" / "v01_mug_724459"
    if not root.exists():
        pytest.skip("dataset not downloaded")
    u = load_usd(root / "v01_mug_724459.usda")
    m = load_mjcf(root / "mjcf" / "v01_mug_724459.xml", compile_model=False)
    b = m.bodies[0]
    assert b.mass == pytest.approx(u.mass, rel=1e-6)
    assert np.allclose(b.com, u.com, atol=1e-7)
    assert np.linalg.norm(b.inertia_tensor - u.inertia_tensor) / np.linalg.norm(u.inertia_tensor) < 1e-5


def test_root_scale_is_kept_in_body_frame(tmp_path):
    path = tmp_path / "scaled.usda"
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/obj")
    st.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(2, 0, 0))
    root.AddScaleOp().Set(Gf.Vec3f(0.25, 0.25, 0.25))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    m = UsdGeom.Mesh.Define(st, "/obj/mesh")
    v, f = box((1.0, 1.0, 1.0))
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(f.ravel().tolist()))
    st.GetRootLayer().Save()
    u = load_usd(path)
    ext = u.visuals[0].vertices.max(0) - u.visuals[0].vertices.min(0)
    assert np.allclose(ext, 0.25, atol=1e-6)
    assert np.allclose(u.visuals[0].vertices.mean(0), 0, atol=1e-6)
