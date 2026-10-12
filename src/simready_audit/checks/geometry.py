import numpy as np
import trimesh
from scipy.spatial import Delaunay

from ..analytics import mesh as M
from .base import FAIL, INFO, PASS, SKIP, WARN, check


@check("B1", "Collider is closed and 2-manifold", "geometry")
def closed(ctx):
    t, _, welded = M.closed_topology(ctx.collider.vertices, ctx.collider.faces)
    ok = t["boundary_edges"] == 0 and t["nonmanifold_edges"] == 0
    return (PASS if ok else FAIL), {"boundary_edges": t["boundary_edges"], "nonmanifold_edges": t["nonmanifold_edges"], "closed_after_merging_coincident_vertices": welded}, ""


@check("B2", "Collider winding consistent and outward", "geometry")
def winding(ctx):
    t, faces, welded = M.closed_topology(ctx.collider.vertices, ctx.collider.faces)
    from ..io.types import TriMesh
    parts = M.components(TriMesh("w", ctx.collider.vertices, faces)) if welded else ctx.collider_parts
    vols = [M.mass_properties(p.vertices, p.faces)["volume"] for p in parts]
    inward = sum(1 for v in vols if v <= 0)
    ok = t["inconsistent_winding_edges"] == 0 and inward == 0
    return (PASS if ok else FAIL), {"inconsistent_edges": t["inconsistent_winding_edges"], "inward_components": inward}, ""


@check("B3", "No degenerate collider faces", "geometry")
def degenerate(ctx):
    a = M.face_areas(ctx.collider.vertices, ctx.collider.faces)
    n = int((a < ctx.cfg["geometry"]["degenerate_face_area_m2"]).sum())
    return (WARN if n else PASS), {"degenerate_faces": n, "faces": len(a)}, ""


@check("B4", "Collider components", "geometry")
def parts(ctx):
    return INFO, {"components": len(ctx.collider_parts), "visual_zones": len(ctx.usd.visuals)}, ""


def _inside_any(points, hulls, skip):
    inside = np.zeros(len(points), dtype=bool)
    for j, h in enumerate(hulls):
        if j == skip:
            continue
        d = Delaunay(h.vertices)
        inside |= d.find_simplex(points) >= 0
    return inside


def surface_distances(target_mesh, pts, threshold, rng, dense=100000):
    from scipy.spatial import cKDTree
    dense_pts = M.sample_surface(target_mesh, dense, rng)
    area = M.face_areas(target_mesh.vertices, target_mesh.faces).sum()
    cover = 3.0 * np.sqrt(area / dense)
    d = cKDTree(dense_pts).query(pts)[0]
    exact_mask = np.zeros(len(d), dtype=bool)
    t = trimesh.Trimesh(target_mesh.vertices, target_mesh.faces, process=False)

    def refine(idx):
        idx = idx[~exact_mask[idx]]
        if len(idx):
            _, ex, _ = trimesh.proximity.closest_point(t, pts[idx])
            d[idx] = np.minimum(d[idx], ex)
            exact_mask[idx] = True

    refine(np.where(d <= threshold + cover)[0])
    for _ in range(20):
        top = np.argsort(-d)[:64]
        top = top[d[top] < 0.1]
        if len(top) == 0 or exact_mask[top[0]]:
            break
        refine(top)
    return d


def phantom_surface(ctx, hulls=None):
    hulls = ctx.effective_hulls if hulls is None else hulls
    g = ctx.cfg["geometry"]
    n = g["surface_samples"]
    areas = np.array([M.face_areas(h.vertices, h.faces).sum() for h in hulls])
    counts = np.maximum(1, np.round(n * areas / areas.sum()).astype(int))
    rng = np.random.default_rng(ctx.seed)
    pts, wts = [], []
    for i, (h, k) in enumerate(zip(hulls, counts)):
        p = M.sample_surface(h, int(k), rng)
        if len(hulls) > 1:
            p = p[~_inside_any(p, hulls, i)]
        pts.append(p)
        wts.append(np.full(len(p), areas[i] / k))
    pts = np.vstack(pts)
    wts = np.concatenate(wts)
    dist = surface_distances(ctx.collider, pts, g["phantom_distance_m"], rng)
    far = dist > g["phantom_distance_m"]
    frac = float(wts[far].sum() / wts.sum())
    return {
        "phantom_area_fraction": frac,
        "p95_mm_upper_bound": float(np.percentile(dist, 95) * 1e3),
        "max_mm": float(dist.max() * 1e3),
        "samples": int(len(pts)),
        "hulls": len(hulls),
    }


@check("B5", "MuJoCo effective collider has no phantom surface", "geometry")
def phantom(ctx):
    if not ctx.effective_hulls:
        return SKIP, {}, "no MJCF mesh collision geoms"
    g = ctx.cfg["geometry"]
    m = phantom_surface(ctx)
    frac = m["phantom_area_fraction"]
    msg = f"{frac:.1%} of the MuJoCo collision surface lies > {g['phantom_distance_m']*1e3:.0f} mm from the authored collider (deepest {m['max_mm']:.1f} mm)"
    if frac > g["phantom_area_fraction_warn"]:
        return WARN, m, msg
    return PASS, m, ""


@check("B6", "MJCF collision geometry sits on the USD collider", "geometry")
def source_identity(ctx):
    b = ctx.mj_body
    if b is None or not b.collision_geoms:
        return SKIP, {}, ""
    from scipy.spatial import cKDTree
    from ..analytics.derived import concat, geom_in_body
    mj = concat([geom_in_body(g) for g in b.collision_geoms if g.mesh is not None], "mjcf")
    d1, _ = cKDTree(ctx.collider.vertices).query(mj.vertices)
    identical = bool(d1.max() <= 1e-5 and len(mj.vertices) == len(ctx.collider.vertices))
    rng = np.random.default_rng(ctx.seed)
    tol = ctx.cfg["geometry"]["mjcf_source_tol_m"]
    dist = surface_distances(ctx.collider, mj.vertices, tol, rng)
    m = {"identical_vertices": identical, "max_vertex_offset_mm": float(dist.max() * 1e3), "mjcf_vertices": len(mj.vertices)}
    if dist.max() > tol:
        return WARN, m, f"MJCF collision vertices sit up to {dist.max()*1e3:.1f} mm off the USD collider surface"
    return PASS, m, ""
