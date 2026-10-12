from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from ..io.types import TriMesh
from ..io.usd_loader import quat_to_matrix
from . import mesh as M


def concat(meshes, name):
    vs, fs, off = [], [], 0
    for m in meshes:
        vs.append(m.vertices)
        fs.append(m.faces + off)
        off += len(m.vertices)
    return TriMesh(name=name, vertices=np.vstack(vs), faces=np.vstack(fs))


def geom_in_body(geom):
    v = geom.mesh.vertices
    if geom.quat is not None:
        v = v @ quat_to_matrix(geom.quat).T
    if geom.pos is not None:
        v = v + geom.pos
    return TriMesh(name=geom.mesh.name, vertices=v, faces=geom.mesh.faces)


@dataclass
class AssetContext:
    usd: object
    mjcf: object
    cfg: dict
    seed: int
    cache: dict = field(default_factory=dict)

    @cached_property
    def collider(self):
        return concat([c.mesh for c in self.usd.colliders], "collider")

    @cached_property
    def collider_parts(self):
        return M.components(self.collider)

    @cached_property
    def collider_union(self):
        u, bad = M.union([self.collider])
        return u

    @cached_property
    def solid(self):
        u = self.collider_union
        return M.mass_properties(u.vertices, u.faces)

    @cached_property
    def shell(self):
        u = self.collider_union
        return M.shell_mass_properties(u.vertices, u.faces)

    @cached_property
    def visual_union(self):
        if not self.usd.visuals:
            return None
        v = concat(self.usd.visuals, "visual")
        if M.edge_topology(v.faces)["boundary_edges"]:
            return None
        u, bad = M.union([v])
        return u if u is not None and not bad else None

    @cached_property
    def mass_models(self):
        out = {"solid_collider": self.solid, "shell_collider": self.shell}
        vu = self.visual_union
        if vu is not None:
            out["solid_visual"] = M.mass_properties(vu.vertices, vu.faces)
            out["shell_visual"] = M.shell_mass_properties(vu.vertices, vu.faces)
        return out

    @cached_property
    def envelope(self):
        return M.convex_hull(self.collider, "envelope")

    @cached_property
    def envelope_volume(self):
        e = self.envelope
        return M.mass_properties(e.vertices, e.faces)["volume"]

    @cached_property
    def material(self):
        mats = [c.material for c in self.usd.colliders if c.material is not None]
        return mats[0] if mats else None

    @cached_property
    def mj_body(self):
        return self.mjcf.bodies[0] if self.mjcf and self.mjcf.bodies else None

    @cached_property
    def effective_hulls(self):
        b = self.mj_body
        if b is None:
            return []
        out = []
        for g in b.collision_geoms:
            if g.kind == "mesh" and g.mesh is not None:
                out.append(M.convex_hull(geom_in_body(g), f"hull({g.mesh.name})"))
        return out

    @cached_property
    def visual_zones(self):
        return {z.name: z for z in self.usd.visuals}

    @cached_property
    def body_zone(self):
        z = self.visual_zones
        return z.get("vis_base__body")

    @cached_property
    def rng(self):
        return np.random.default_rng(self.seed)
