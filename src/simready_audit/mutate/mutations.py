import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

from ..fixes.mjcf_rewrite import _abs_assets, write_obj
from ..io.mjcf_loader import read_obj


class Workspace:
    def __init__(self, usda, mjcf, out_dir):
        self.usda_src, self.mjcf_src = Path(usda), Path(mjcf)
        cls = self.usda_src.parent.parent.name
        self.out = Path(out_dir) / "assets" / cls / self.usda_src.stem
        self.out.mkdir(parents=True, exist_ok=True)
        self.usda = self.out / self.usda_src.name
        self.stage = Usd.Stage.Open(str(self.usda_src))
        self.tree = ET.parse(self.mjcf_src)
        self.root = self.tree.getroot()
        _abs_assets(self.root, self.mjcf_src.parent)
        self.mjcf = self.out / "mjcf" / self.mjcf_src.name

    def prim(self):
        return self.stage.GetDefaultPrim()

    def meshes(self, collider_only=False):
        for p in self.stage.Traverse():
            if p.IsA(UsdGeom.Mesh) and (not collider_only or "PhysicsCollisionAPI" in p.GetAppliedSchemas()):
                yield UsdGeom.Mesh(p)

    def collision_obj_elems(self):
        names = {g.get("mesh") for g in self.root.iter("geom") if g.get("class") == "collision"}
        return [m for m in self.root.iter("mesh") if m.get("name") in names]

    def all_obj_elems(self):
        return list(self.root.iter("mesh"))

    def rewrite_obj(self, elem, fn, tag):
        v, f = read_obj(elem.get("file"))
        v, f = fn(v, f)
        dst = self.out / "mjcf" / f"{tag}_{Path(elem.get('file')).name}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        write_obj(dst, v, f)
        elem.set("file", str(dst))

    def inertial(self):
        return self.root.find("worldbody").find("body").find("inertial")

    def save(self):
        self.stage.GetRootLayer().Export(str(self.usda))
        self.mjcf.parent.mkdir(parents=True, exist_ok=True)
        self.tree.write(self.mjcf, encoding="utf-8", xml_declaration=True)
        return self.usda, self.mjcf


def _scale_points(ws, s):
    for m in ws.meshes():
        p = np.asarray(m.GetPointsAttr().Get()) * s
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
    for e in ws.all_obj_elems():
        ws.rewrite_obj(e, lambda v, f: (v * s, f), f"s{s}")


def scale(ws, s):
    _scale_points(ws, s)


def meters_per_unit(ws, mpu):
    UsdGeom.SetStageMetersPerUnit(ws.stage, mpu)


def up_axis_y(ws, _=None):
    UsdGeom.SetStageUpAxis(ws.stage, UsdGeom.Tokens.y)


def _collider_mesh_edit(ws, fn):
    for m in ws.meshes(collider_only=True):
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
        pts, idx = fn(pts, idx)
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        m.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(idx)))
        m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(idx.ravel().tolist()))


def delete_faces(ws, frac, seed=0):
    rng = np.random.default_rng(seed)
    def fn(p, f):
        k = max(1, int(round(frac * len(f))))
        keep = np.ones(len(f), bool)
        keep[rng.choice(len(f), k, replace=False)] = False
        return p, f[keep]
    _collider_mesh_edit(ws, fn)


def flip_faces(ws, frac, seed=0):
    rng = np.random.default_rng(seed)
    def fn(p, f):
        f = f.copy()
        k = max(1, int(round(frac * len(f))))
        sel = rng.choice(len(f), k, replace=False)
        f[sel] = f[sel][:, ::-1]
        return p, f
    _collider_mesh_edit(ws, fn)


def degenerate_faces(ws, n):
    def fn(p, f):
        extra = np.array([[f[i, 0], f[i, 0], f[i, 1]] for i in range(int(n))])
        return p, np.vstack([f, extra])
    _collider_mesh_edit(ws, fn)


def _usd_mass(ws, k):
    a = ws.prim().GetAttribute("physics:mass")
    a.Set(float(a.Get()) * k)
    d = ws.prim().GetAttribute("physics:diagonalInertia")
    d.Set(Gf.Vec3f(*(np.asarray(d.Get()) * k).tolist()))


def mass_usd_only(ws, k):
    _usd_mass(ws, k)


def mass_both(ws, k):
    _usd_mass(ws, k)
    i = ws.inertial()
    i.set("mass", f"{float(i.get('mass')) * k:.9g}")
    i.set("fullinertia", " ".join(f"{float(x) * k:.9g}" for x in i.get("fullinertia").split()))


def com_offset_both(ws, d):
    a = ws.prim().GetAttribute("physics:centerOfMass")
    a.Set(Gf.Vec3f(*(np.asarray(a.Get()) + [d, 0, 0]).tolist()))
    i = ws.inertial()
    pos = np.array([float(x) for x in i.get("pos").split()]) + [d, 0, 0]
    i.set("pos", " ".join(f"{x:.9g}" for x in pos))


def com_offset_mjcf(ws, d):
    i = ws.inertial()
    pos = np.array([float(x) for x in i.get("pos").split()]) + [d, 0, 0]
    i.set("pos", " ".join(f"{x:.9g}" for x in pos))


def inertia_scale_both(ws, k):
    d = ws.prim().GetAttribute("physics:diagonalInertia")
    d.Set(Gf.Vec3f(*(np.asarray(d.Get()) * k).tolist()))
    i = ws.inertial()
    i.set("fullinertia", " ".join(f"{float(x) * k:.9g}" for x in i.get("fullinertia").split()))


def inertia_mjcf_only(ws, k):
    i = ws.inertial()
    vals = [float(x) for x in i.get("fullinertia").split()]
    vals[0] *= k
    i.set("fullinertia", " ".join(f"{x:.9g}" for x in vals))


def friction_mjcf(ws, mu):
    for d in ws.root.iter("default"):
        if d.get("class") == "collision":
            g = d.find("geom")
            vals = g.get("friction").split()
            vals[0] = f"{mu:.6g}"
            g.set("friction", " ".join(vals))


def collider_offset_mjcf(ws, d):
    for e in ws.collision_obj_elems():
        ws.rewrite_obj(e, lambda v, f: (v + [d, 0, 0], f), f"o{d}")


def max_mass_equals_mass(ws, _=None):
    p = ws.prim()
    m = float(p.GetAttribute("physics:mass").Get())
    p.SetCustomDataByKey("kaedim:max_mass_kg", round(m, 4))


def restitution_invalid(ws, e):
    for p in ws.stage.Traverse():
        if "PhysicsMaterialAPI" in p.GetAppliedSchemas():
            p.GetAttribute("physics:restitution").Set(float(e))


def unbind_material(ws, _=None):
    for p in ws.stage.Traverse():
        if "PhysicsCollisionAPI" in p.GetAppliedSchemas():
            p.GetRelationship("material:binding:physics").ClearTargets(True)


MUTATIONS = {
    "scale_x10": (scale, 10.0, ["C1", "C2"]),
    "scale_x0.1": (scale, 0.1, ["C1", "C2"]),
    "units_cm": (meters_per_unit, 0.01, ["A1"]),
    "up_axis_y": (up_axis_y, None, ["A1"]),
    "delete_1pct_faces": (delete_faces, 0.01, ["B1"]),
    "flip_10pct_faces": (flip_faces, 0.10, ["B2"]),
    "flip_all_faces": (flip_faces, 1.0, ["B2"]),
    "degenerate_faces": (degenerate_faces, 5, ["B3"]),
    "mass_x10_usd_only": (mass_usd_only, 10.0, ["A4", "A6"]),
    "mass_x2_both": (mass_both, 2.0, ["C2"]),
    "com_offset_both_10mm": (com_offset_both, 0.010, ["C6"]),
    "com_offset_mjcf_1mm": (com_offset_mjcf, 0.001, ["A5"]),
    "inertia_x2_both": (inertia_scale_both, 2.0, ["C5"]),
    "inertia_mjcf_only_x1.1": (inertia_mjcf_only, 1.1, ["A6"]),
    "friction_mjcf_0.9": (friction_mjcf, 0.9, ["A7"]),
    "collider_offset_mjcf_5mm": (collider_offset_mjcf, 0.005, ["B6"]),
    "max_mass_equals_mass": (max_mass_equals_mass, None, ["A10"]),
    "restitution_1.5": (restitution_invalid, 1.5, ["A2"]),
    "unbind_physics_material": (unbind_material, None, ["A2"]),
}

SWEEPS = {
    "scale": (scale, [1.02, 1.05, 1.1, 1.25, 2.0], ["C1", "C2"]),
    "com_offset_both": (com_offset_both, [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02], ["C6"]),
    "mass_both": (mass_both, [1.02, 1.05, 1.1, 1.2, 1.5], ["C2"]),
    "inertia_scale_both": (inertia_scale_both, [1.05, 1.1, 1.2, 1.5], ["C5"]),
    "collider_offset_mjcf": (collider_offset_mjcf, [0.0005, 0.0015, 0.003, 0.005], ["B6"]),
}


def materialize(usda, mjcf, out_dir, fn, arg):
    ws = Workspace(usda, mjcf, out_dir)
    if fn is not None:
        fn(ws, arg)
    return ws.save()


def rotate_z(usda, mjcf, out_dir, degrees):
    ws = Workspace(usda, mjcf, out_dir)
    t = np.radians(degrees)
    r = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    for m in ws.meshes():
        p = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64) @ r.T
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
    for e in ws.all_obj_elems():
        ws.rewrite_obj(e, lambda v, f: (v @ r.T, f), "rz")
    prim = ws.prim()
    c = prim.GetAttribute("physics:centerOfMass")
    c.Set(Gf.Vec3f(*(np.asarray(c.Get(), dtype=np.float64) @ r.T).tolist()))
    q = prim.GetAttribute("physics:principalAxes").Get()
    rq = Gf.Quatf(float(np.cos(t / 2)), Gf.Vec3f(0, 0, float(np.sin(t / 2))))
    prim.GetAttribute("physics:principalAxes").Set(rq * q)
    i = ws.inertial()
    pos = np.array([float(x) for x in i.get("pos").split()]) @ r.T
    i.set("pos", " ".join(f"{x:.9g}" for x in pos))
    ixx, iyy, izz, ixy, ixz, iyz = (float(x) for x in i.get("fullinertia").split())
    full = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    rot = r @ full @ r.T
    i.set("fullinertia", " ".join(f"{x:.12g}" for x in (rot[0, 0], rot[1, 1], rot[2, 2], rot[0, 1], rot[0, 2], rot[1, 2])))
    return ws.save()


def permute_faces(usda, mjcf, out_dir, seed=0):
    ws = Workspace(usda, mjcf, out_dir)
    rng = np.random.default_rng(seed)
    for m in ws.meshes():
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        pts = np.asarray(m.GetPointsAttr().Get())
        perm_v = rng.permutation(len(pts))
        inv = np.empty_like(perm_v)
        inv[perm_v] = np.arange(len(perm_v))
        idx = inv[idx][rng.permutation(len(idx))]
        idx = np.roll(idx, 1, axis=1)
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts[perm_v]))
        m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(idx.ravel().tolist()))
    for e in ws.all_obj_elems():
        def fn(v, f):
            perm_v = rng.permutation(len(v))
            inv = np.empty_like(perm_v)
            inv[perm_v] = np.arange(len(perm_v))
            return v[perm_v], inv[f][rng.permutation(len(f))]
        ws.rewrite_obj(e, fn, "perm")
    return ws.save()


def root_translate(usda, mjcf, out_dir, offset=(0.3, -0.2, 0.1)):
    ws = Workspace(usda, mjcf, out_dir)
    x = UsdGeom.Xformable(ws.prim())
    ops = [o for o in x.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    op = ops[0] if ops else x.AddTranslateOp()
    op.Set(Gf.Vec3d(*offset))
    ws.root.find("worldbody").find("body").set("pos", " ".join(map(str, offset)))
    return ws.save()


def to_usdc(usda, mjcf, out_dir):
    ws = Workspace(usda, mjcf, out_dir)
    ws.usda = ws.out / (ws.usda_src.stem + ".usdc")
    return ws.save()


INVARIANCES = {
    "rotate_z_37deg": (rotate_z, 37.0),
    "permute_vertices_and_faces": (permute_faces, 0),
    "root_translate": (root_translate, (0.3, -0.2, 0.1)),
    "binary_usdc": (to_usdc, None),
}
