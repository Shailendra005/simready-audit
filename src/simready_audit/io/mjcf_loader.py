import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .types import MjcfAsset, MjcfBody, MjcfGeom, TriMesh


def _floats(s):
    return None if s is None else np.array([float(v) for v in s.split()], dtype=np.float64)


def read_obj(path):
    vertices = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3), np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def full_inertia_matrix(vals):
    ixx, iyy, izz, ixy, ixz, iyz = vals
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)


def _defaults(root):
    out = {}
    for d in root.iter("default"):
        cls = d.get("class")
        g = d.find("geom")
        if cls and g is not None:
            out[cls] = dict(g.attrib)
    return out


def load_mjcf(path, compile_model=True):
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    base = path.parent
    compiler = root.find("compiler")
    meshdir = base / (compiler.get("meshdir", "") if compiler is not None else "")
    mesh_files = {}
    for asset in root.iter("asset"):
        for m in asset.findall("mesh"):
            mesh_files[m.get("name")] = m
    textures = [t.get("file") for t in root.iter("texture") if t.get("file")]
    option = root.find("option")
    defaults = _defaults(root)
    bodies = []
    has_free = any(True for _ in root.iter("freejoint")) or any(j.get("type") == "free" for j in root.iter("joint"))

    def walk(elem, parent):
        for b in elem.findall("body"):
            inert = b.find("inertial")
            tensor = None
            mass = com = None
            if inert is not None:
                mass = float(inert.get("mass")) if inert.get("mass") else None
                com = _floats(inert.get("pos"))
                if inert.get("fullinertia"):
                    tensor = full_inertia_matrix(_floats(inert.get("fullinertia")))
                elif inert.get("diaginertia"):
                    tensor = np.diag(_floats(inert.get("diaginertia")))
                    if inert.get("quat"):
                        from .usd_loader import quat_to_matrix
                        r = quat_to_matrix(_floats(inert.get("quat")))
                        tensor = r @ tensor @ r.T
            joints = [dict(j.attrib) for j in b.findall("joint")] + [{"type": "free"} for _ in b.findall("freejoint")]
            cgeoms, vgeoms = [], []
            for g in b.findall("geom"):
                attrs = dict(defaults.get(g.get("class"), {}))
                attrs.update(g.attrib)
                mesh_name = attrs.get("mesh")
                mesh_file = None
                tri = None
                if mesh_name and mesh_name in mesh_files:
                    mesh_file = str(meshdir / mesh_files[mesh_name].get("file"))
                    v, f = read_obj(mesh_file)
                    scale = _floats(mesh_files[mesh_name].get("scale"))
                    if scale is not None:
                        v = v * scale
                    tri = TriMesh(name=mesh_name, vertices=v, faces=f)
                geom = MjcfGeom(
                    name=attrs.get("name"),
                    kind=attrs.get("type", "sphere"),
                    mesh_file=mesh_file,
                    mesh=tri,
                    friction=_floats(attrs.get("friction")),
                    solref=_floats(attrs.get("solref")),
                    solimp=_floats(attrs.get("solimp")),
                    condim=int(attrs["condim"]) if "condim" in attrs else None,
                    priority=int(attrs["priority"]) if "priority" in attrs else None,
                    margin=float(attrs["margin"]) if "margin" in attrs else None,
                    contype=int(attrs.get("contype", 1)),
                    conaffinity=int(attrs.get("conaffinity", 1)),
                    group=int(attrs["group"]) if "group" in attrs else None,
                    pos=_floats(attrs.get("pos")),
                    quat=_floats(attrs.get("quat")),
                )
                (cgeoms if geom.contype or geom.conaffinity else vgeoms).append(geom)
            bodies.append(MjcfBody(
                name=b.get("name"), parent=parent, pos=_floats(b.get("pos")) if b.get("pos") else np.zeros(3),
                mass=mass, com=com, inertia_tensor=tensor, joints=joints,
                collision_geoms=cgeoms, visual_geoms=vgeoms,
            ))
            walk(b, b.get("name"))

    walk(root.find("worldbody"), None)
    error = None
    if compile_model:
        try:
            mujoco.MjModel.from_xml_path(str(path))
        except Exception as e:
            error = str(e)
    vid = path.stem
    return MjcfAsset(
        path=str(path),
        vid=vid,
        cls=path.parent.parent.parent.name,
        timestep=float(option.get("timestep", 0.002)) if option is not None else 0.002,
        integrator=option.get("integrator", "Euler") if option is not None else "Euler",
        cone=option.get("cone", "pyramidal") if option is not None else "pyramidal",
        bodies=bodies,
        has_freejoint=has_free,
        texture_files=textures,
        compile_error=error,
    )
