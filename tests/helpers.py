import numpy as np
import trimesh


def tri(m):
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def box(extents, center=(0, 0, 0)):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(center)
    return tri(m)


def cylinder(r, h, sections=256, z0=0.0):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=sections)
    m.apply_translation([0, 0, z0 + h / 2])
    return tri(m)


def cup_boxes(outer=0.08, height=0.09, wall=0.006):
    parts = [box((outer, outer, wall), (0, 0, wall / 2))]
    half = outer / 2 - wall / 2
    for dx, dy, ex, ey in ((half, 0, wall, outer), (-half, 0, wall, outer), (0, half, outer, wall), (0, -half, outer, wall)):
        parts.append(box((ex, ey, height), (dx, dy, height / 2)))
    return parts


def merge(parts):
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.vstack(vs), np.vstack(fs)
