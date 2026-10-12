from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .types import Collider, PhysicsMaterial, TriMesh, UsdAsset


def _get(prim, name):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.HasAuthoredValue() else None


def _triangulate(counts, indices):
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if len(counts) and np.all(counts == 3):
        return indices.reshape(-1, 3)
    tris = []
    offset = 0
    for c in counts:
        poly = indices[offset:offset + c]
        for k in range(1, c - 1):
            tris.append((poly[0], poly[k], poly[k + 1]))
        offset += c
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def _up_axis_rotation(up_axis):
    if up_axis == "Y":
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return np.eye(3)


def quat_to_matrix(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _read_material(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim or "PhysicsMaterialAPI" not in prim.GetAppliedSchemas():
        return None
    return PhysicsMaterial(
        path=str(path),
        density=_get(prim, "physics:density"),
        static_friction=_get(prim, "physics:staticFriction"),
        dynamic_friction=_get(prim, "physics:dynamicFriction"),
        restitution=_get(prim, "physics:restitution"),
        friction_combine=_get(prim, "physxMaterial:frictionCombineMode"),
        restitution_combine=_get(prim, "physxMaterial:restitutionCombineMode"),
    )


def _rigid_part(m):
    lin = m[:3, :3]
    u, _, vt = np.linalg.svd(lin)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    out = np.eye(4)
    out[:3, :3] = r
    out[3, :3] = m[3, :3]
    return out


def load_usd(path):
    path = Path(path)
    stage = Usd.Stage.Open(str(path))
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    kpu = UsdPhysics.GetStageKilogramsPerUnit(stage)
    up = str(UsdGeom.GetStageUpAxis(stage))
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot().GetChildren()[0]
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_world = np.array(cache.GetLocalToWorldTransform(root), dtype=np.float64)
    root_rigid = _rigid_part(root_world)
    root_inv = np.linalg.inv(root_rigid)
    root_linear = root_world[:3, :3] @ np.linalg.inv(root_rigid[:3, :3])
    up_rot = _up_axis_rotation(up)

    def to_body(points, prim):
        m = np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64) @ root_inv
        p = np.asarray(points, dtype=np.float64)
        ph = np.c_[p, np.ones(len(p))] @ m
        return (ph[:, :3] * mpu) @ up_rot.T

    mass = _get(root, "physics:mass")
    com = _get(root, "physics:centerOfMass")
    diag = _get(root, "physics:diagonalInertia")
    axes = _get(root, "physics:principalAxes")
    mass = None if mass is None else float(mass) * kpu
    com = None if com is None else ((np.array(com, dtype=np.float64) @ root_linear) * mpu) @ up_rot.T
    diag_arr = None if diag is None else np.array(diag, dtype=np.float64) * kpu * mpu * mpu
    if diag_arr is not None and not np.any(diag_arr):
        diag_arr = None
    axes_arr = None
    tensor = None
    if axes is not None:
        q = Gf.Quatd(axes)
        axes_arr = np.array([q.GetReal(), *q.GetImaginary()], dtype=np.float64)
    if diag_arr is not None:
        r = quat_to_matrix(axes_arr) if axes_arr is not None else np.eye(3)
        local = r @ np.diag(diag_arr) @ r.T
        stretched = root_linear.T @ local @ root_linear
        tensor = up_rot @ stretched @ up_rot.T

    custom = dict(root.GetCustomData())
    asset = UsdAsset(
        path=str(path),
        vid=path.stem,
        cls=path.parent.parent.name,
        meters_per_unit=float(mpu),
        kilograms_per_unit=float(kpu),
        up_axis=up,
        root_path=str(root.GetPath()),
        applied_schemas=list(root.GetAppliedSchemas()),
        mass=mass,
        com=com,
        diagonal_inertia=diag_arr,
        principal_axes=axes_arr,
        inertia_tensor=tensor,
        custom={k: v for k, v in custom.items() if k != "userDocBrief"},
    )

    for prim in stage.Traverse():
        schemas = prim.GetAppliedSchemas()
        if "PhysicsMaterialAPI" in schemas:
            asset.materials.append(_read_material(stage, prim.GetPath()))
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        faces = _triangulate(mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get())
        tri = TriMesh(name=prim.GetName(), vertices=to_body(mesh.GetPointsAttr().Get(), prim), faces=faces)
        if "PhysicsCollisionAPI" in schemas:
            material = None
            for relname in ("material:binding:physics", "material:binding"):
                rel = prim.GetRelationship(relname)
                targets = rel.GetTargets() if rel else []
                material = _read_material(stage, targets[0]) if targets else None
                if material is not None:
                    break
            sdf_res = _get(prim, "physxSDFMeshCollision:sdfResolution")
            asset.colliders.append(Collider(
                name=prim.GetName(),
                mesh=tri,
                approximation=_get(prim, "physics:approximation"),
                sdf_resolution=None if sdf_res is None else int(sdf_res),
                material=material,
            ))
        else:
            asset.visuals.append(tri)
    return asset
