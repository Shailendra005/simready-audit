import manifold3d as mf
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

from ..io.types import TriMesh


def directed_edges(faces):
    return np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]


def edge_topology(faces):
    d = directed_edges(faces)
    und = np.sort(d, axis=1)
    _, counts = np.unique(und, axis=0, return_counts=True)
    _, dcounts = np.unique(d, axis=0, return_counts=True)
    return {
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "inconsistent_winding_edges": int((dcounts > 1).sum()),
        "edges": int(len(counts)),
    }


def weld(vertices, faces, tol=1e-9):
    from scipy.spatial import cKDTree
    pairs = cKDTree(vertices).query_pairs(tol, output_type="ndarray")
    parent = np.arange(len(vertices))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(len(vertices))])
    f = roots[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return f[keep]


def closed_topology(vertices, faces):
    raw = edge_topology(faces)
    if raw["boundary_edges"] == 0 and raw["nonmanifold_edges"] == 0:
        return raw, faces, False
    welded = weld(vertices, faces)
    w = edge_topology(welded)
    if w["boundary_edges"] == 0 and w["nonmanifold_edges"] == 0:
        return w, welded, True
    return raw, faces, False


def face_areas(vertices, faces):
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def components(mesh):
    v, f = mesh.vertices, mesh.faces
    n = len(v)
    r = np.r_[f[:, 0], f[:, 1], f[:, 2]]
    c = np.r_[f[:, 1], f[:, 2], f[:, 0]]
    _, labels = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
    out = []
    face_labels = labels[f[:, 0]]
    for i, lab in enumerate(np.unique(face_labels)):
        fi = f[face_labels == lab]
        used = np.unique(fi)
        remap = -np.ones(n, dtype=np.int64)
        remap[used] = np.arange(len(used))
        out.append(TriMesh(name=f"{mesh.name}#{i}", vertices=v[used], faces=remap[fi]))
    return out


def mass_properties(vertices, faces, density=1.0):
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    vol6 = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
    volume = vol6.sum() / 6.0
    if abs(volume) < 1e-18:
        return {"volume": 0.0, "mass": 0.0, "com": np.zeros(3), "inertia": np.zeros((3, 3))}
    com = (vol6[:, None] * (v0 + v1 + v2)).sum(0) / (24.0 * volume)
    def s(a, b):
        return (
            2 * v0[:, a] * v0[:, b] + 2 * v1[:, a] * v1[:, b] + 2 * v2[:, a] * v2[:, b]
            + v0[:, a] * v1[:, b] + v1[:, a] * v0[:, b]
            + v0[:, a] * v2[:, b] + v2[:, a] * v0[:, b]
            + v1[:, a] * v2[:, b] + v2[:, a] * v1[:, b]
        )
    m2 = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            m2[a, b] = (vol6 * s(a, b)).sum() / 120.0
    mass = density * volume
    second = density * m2
    origin = np.trace(second) * np.eye(3) - second
    shifted = origin - mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    return {"volume": float(volume), "mass": float(mass), "com": com, "inertia": shifted, "area": float(0.5 * np.linalg.norm(cross, axis=1).sum())}


def shell_mass_properties(vertices, faces, areal_density=1.0):
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    mass = areal_density * areas.sum()
    com = (areas[:, None] * (a + b + c) / 3.0).sum(0) * areal_density / mass
    second = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            second[i, j] = (areas * (
                a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j]
                + (a[:, i] + b[:, i] + c[:, i]) * (a[:, j] + b[:, j] + c[:, j])
            ) / 12.0).sum() * areal_density
    origin = np.trace(second) * np.eye(3) - second
    shifted = origin - mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    return {"area": float(areas.sum()), "mass": float(mass), "com": com, "inertia": shifted}


def to_manifold(mesh):
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float64).astype(np.float32), tri_verts=np.asarray(mesh.faces, np.uint32)))


def manifold_to_trimesh(m, name):
    out = m.to_mesh()
    return TriMesh(name=name, vertices=np.asarray(out.vert_properties[:, :3], np.float64), faces=np.asarray(out.tri_verts, np.int64))


def union(meshes, name="union"):
    parts = []
    for m in meshes:
        parts.extend(components(m))
    mans = [to_manifold(p) for p in parts]
    bad = [p.name for p, m in zip(parts, mans) if m.status() != mf.Error.NoError or m.is_empty()]
    mans = [m for m in mans if m.status() == mf.Error.NoError and not m.is_empty()]
    if not mans:
        return None, bad
    u = mf.Manifold.batch_boolean(mans, mf.OpType.Add) if len(mans) > 1 else mans[0]
    return manifold_to_trimesh(u, name), bad


def convex_hull(mesh, name=None):
    h = ConvexHull(mesh.vertices)
    faces = h.simplices.copy()
    centroid = mesh.vertices[h.vertices].mean(0)
    v = mesh.vertices
    n = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    flip = np.einsum("ij,ij->i", n, v[faces[:, 0]] - centroid) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    used = np.unique(faces)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return TriMesh(name=name or f"hull({mesh.name})", vertices=v[used], faces=remap[faces])


def bbox(mesh):
    return mesh.vertices.min(0), mesh.vertices.max(0)


def sample_surface(mesh, n, rng):
    areas = face_areas(mesh.vertices, mesh.faces)
    idx = rng.choice(len(areas), size=n, p=areas / areas.sum())
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    a, b, c = (mesh.vertices[mesh.faces[idx, k]] for k in range(3))
    return (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c
