import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .analytics import mesh as M
from .fixes.mjcf_rewrite import write_obj


def _principal(inertia):
    w, v = np.linalg.eigh(0.5 * (inertia + inertia.T))
    if np.linalg.det(v) < 0:
        v[:, 0] = -v[:, 0]
    q = Gf.Quatd(Gf.Matrix3d(*v.T.ravel().tolist()).ExtractRotation().GetQuat())
    return w, q


def write_usd(path, name, visuals, collider, mass, com, inertia, density=1000.0, sf=0.5, df=0.4, e=0.1, approximation="convexHull", max_mass=None, meters_per_unit=1.0, up_axis="Z"):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mapi = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mapi.CreateMassAttr(float(mass))
    mapi.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    w, q = _principal(np.asarray(inertia))
    mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, w)))
    mapi.CreatePrincipalAxesAttr(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
    root.GetPrim().SetCustomDataByKey("kaedim:mass_kg", float(mass))
    root.GetPrim().SetCustomDataByKey("kaedim:max_mass_kg", float(max_mass if max_mass is not None else mass * 2))
    mat = UsdShade.Material.Define(stage, f"/{name}/Physics/PhysicsMaterial_body")
    papi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    papi.CreateDensityAttr(float(density))
    papi.CreateStaticFrictionAttr(float(sf))
    papi.CreateDynamicFrictionAttr(float(df))
    papi.CreateRestitutionAttr(float(e))
    for vname, (v, f) in visuals.items():
        m = UsdGeom.Mesh.Define(stage, f"/{name}/{vname}")
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(v, np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(np.asarray(f).ravel().tolist()))
    c = UsdGeom.Mesh.Define(stage, f"/{name}/collision")
    v, f = collider
    c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(v, np.float32)))
    c.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
    c.CreateFaceVertexIndicesAttr(Vt.IntArray(np.asarray(f).ravel().tolist()))
    c.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(c.GetPrim()).CreateApproximationAttr(approximation)
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()
    return Path(path)


def write_mjcf(path, name, visuals, collision_hulls, mass, com, inertia, friction=0.35, solref=(0.004, 1.0)):
    path = Path(path)
    (path.parent / "collision").mkdir(parents=True, exist_ok=True)
    (path.parent / "visuals").mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "compiler", angle="degree", autolimits="true")
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast")
    d = ET.SubElement(root, "default")
    dv = ET.SubElement(d, "default", {"class": "visual"})
    ET.SubElement(dv, "geom", group="2", contype="0", conaffinity="0")
    dc = ET.SubElement(d, "default", {"class": "collision"})
    ET.SubElement(dc, "geom", group="3", contype="1", conaffinity="1", friction=f"{friction} 0.005 0.0001", solref=f"{solref[0]} {solref[1]}", priority="1")
    asset = ET.SubElement(root, "asset")
    wb = ET.SubElement(root, "worldbody")
    body = ET.SubElement(wb, "body", name="base", pos="0 0 0")
    i = np.asarray(inertia)
    ET.SubElement(body, "inertial", pos=" ".join(f"{x:.9g}" for x in com), mass=f"{mass:.9g}", fullinertia=" ".join(f"{x:.9g}" for x in (i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])))
    for vname, (v, f) in visuals.items():
        write_obj(path.parent / "visuals" / f"{vname}.obj", v, f)
        ET.SubElement(asset, "mesh", name=vname, file=f"visuals/{vname}.obj")
        ET.SubElement(body, "geom", {"class": "visual", "type": "mesh", "mesh": vname, "name": vname})
    for k, (v, f) in enumerate(collision_hulls):
        n = f"col_base_{k:03d}"
        write_obj(path.parent / "collision" / f"{n}.obj", v, f)
        ET.SubElement(asset, "mesh", name=n, file=f"collision/{n}.obj")
        ET.SubElement(body, "geom", {"class": "collision", "type": "mesh", "mesh": n})
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def write_asset(out_dir, name, cls, visuals, collider, density=1000.0, mjcf_hulls=None, **kw):
    out = Path(out_dir) / "assets" / cls / name
    (out / "mjcf").mkdir(parents=True, exist_ok=True)
    v, f = collider
    props = M.mass_properties(np.asarray(v, float), np.asarray(f), density)
    mass = kw.pop("mass", props["mass"])
    com = kw.pop("com", props["com"])
    inertia = kw.pop("inertia", props["inertia"] * mass / props["mass"])
    usd_kw = {k: kw[k] for k in ("sf", "df", "e", "approximation", "max_mass", "meters_per_unit", "up_axis") if k in kw}
    mj_kw = {k: kw[k] for k in ("friction", "solref") if k in kw}
    usda = write_usd(out / f"{name}.usda", name, visuals, collider, mass, com, inertia, density=density, **usd_kw)
    hulls = mjcf_hulls if mjcf_hulls is not None else [collider]
    mjcf = write_mjcf(out / "mjcf" / f"{name}.xml", name, visuals, hulls, mass, com, inertia, **mj_kw)
    return {"vid": name, "cls": cls, "usda": usda, "mjcf": mjcf}
