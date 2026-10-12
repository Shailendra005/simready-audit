import math

import mujoco
import numpy as np
import trimesh

from .scene import add_floor, load_spec, strip_visuals


def opening_geometry(body_zone):
    t = trimesh.Trimesh(body_zone.vertices, body_zone.faces, process=False)
    lo, hi = body_zone.vertices.min(0), body_zone.vertices.max(0)
    c = 0.5 * (lo + hi)
    z_rim = float(hi[2])
    z_probe = z_rim - 0.01
    dirs = np.array([[math.cos(a), math.sin(a), 0.0] for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)])
    origins = np.tile([c[0], c[1], z_probe], (len(dirs), 1))
    locs, ray_idx, _ = t.ray.intersects_location(origins, dirs, multiple_hits=False)
    if len(locs) == 0:
        return None
    r_inner = float(np.min(np.linalg.norm(locs[:, :2] - c[:2], axis=1)))
    down, _, _ = t.ray.intersects_location([[c[0], c[1], z_rim + 1e-3]], [[0, 0, -1.0]], multiple_hits=False)
    z_floor = float(down[0][2]) if len(down) else float(lo[2])
    return {"center": c[:2].tolist(), "z_rim": z_rim, "r_inner": r_inner, "z_floor": z_floor, "depth": z_rim - z_floor}


def cavity_probe(mjcf_path, body_zone, cfg, spec=None, duration=1.5):
    s = cfg["sim"]
    geo = opening_geometry(body_zone)
    if geo is None or geo["depth"] < 0.005:
        return {"result": "NO_CAVITY", **(geo or {})}
    r = float(np.clip(s["probe_radius_frac"] * geo["r_inner"], s["probe_radius_min_m"], s["probe_radius_max_m"]))
    spec = spec or load_spec(mjcf_path)
    strip_visuals(spec)
    add_floor(spec)
    probe = spec.worldbody.add_body()
    probe.name = "probe"
    probe.pos = [geo["center"][0], geo["center"][1], geo["z_rim"] + s["probe_drop_above_rim_m"] + r]
    probe.add_freejoint()
    g = probe.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [r, 0, 0]
    g.mass = 0.005
    model = spec.compile()
    data = mujoco.MjData(model)
    bid = model.body("probe").id
    for _ in range(int(duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
    z = float(data.xpos[bid][2])
    xy = float(np.linalg.norm(data.xpos[bid][:2] - np.asarray(geo["center"])))
    inside = xy < geo["r_inner"]
    if not np.isfinite(z):
        result = "UNSTABLE"
    elif z < geo["z_rim"] - r and inside:
        result = "ENTERED"
    elif z >= geo["z_rim"]:
        result = "BLOCKED"
    else:
        result = "OTHER"
    return {
        "result": result,
        "probe_radius_mm": r * 1e3,
        "final_z_mm": z * 1e3,
        "final_xy_from_centre_mm": xy * 1e3,
        "rim_z_mm": geo["z_rim"] * 1e3,
        "floor_z_mm": geo["z_floor"] * 1e3,
        "reached_floor": bool(abs(z - (geo["z_floor"] + r)) <= 0.003),
        "inner_radius_mm": geo["r_inner"] * 1e3,
        "depth_mm": geo["depth"] * 1e3,
    }
