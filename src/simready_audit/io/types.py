from dataclasses import dataclass, field

import numpy as np


@dataclass
class TriMesh:
    name: str
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def n_faces(self):
        return len(self.faces)


@dataclass
class PhysicsMaterial:
    path: str
    density: float | None
    static_friction: float | None
    dynamic_friction: float | None
    restitution: float | None
    friction_combine: str | None
    restitution_combine: str | None


@dataclass
class Collider:
    name: str
    mesh: TriMesh
    approximation: str | None
    sdf_resolution: int | None
    material: PhysicsMaterial | None


@dataclass
class UsdAsset:
    path: str
    vid: str
    cls: str
    meters_per_unit: float
    kilograms_per_unit: float
    up_axis: str
    root_path: str
    applied_schemas: list
    mass: float | None
    com: np.ndarray | None
    diagonal_inertia: np.ndarray | None
    principal_axes: np.ndarray | None
    inertia_tensor: np.ndarray | None
    custom: dict
    visuals: list = field(default_factory=list)
    colliders: list = field(default_factory=list)
    materials: list = field(default_factory=list)


@dataclass
class MjcfGeom:
    name: str | None
    kind: str
    mesh_file: str | None
    mesh: TriMesh | None
    friction: np.ndarray | None
    solref: np.ndarray | None
    solimp: np.ndarray | None
    condim: int | None
    priority: int | None
    margin: float | None
    contype: int
    conaffinity: int
    group: int | None
    pos: np.ndarray | None = None
    quat: np.ndarray | None = None


@dataclass
class MjcfBody:
    name: str
    parent: str | None
    pos: np.ndarray
    mass: float | None
    com: np.ndarray | None
    inertia_tensor: np.ndarray | None
    joints: list
    collision_geoms: list
    visual_geoms: list


@dataclass
class MjcfAsset:
    path: str
    vid: str
    cls: str
    timestep: float
    integrator: str
    cone: str
    bodies: list
    has_freejoint: bool
    texture_files: list
    compile_error: str | None
