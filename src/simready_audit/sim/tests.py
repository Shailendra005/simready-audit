import math

import mujoco
import numpy as np

from .scene import collision_points, free_scene, quat_from_axis_angle, rotmat, slide_scene, total_mass

AXIS_ORIENTATIONS = {
    "upright": (np.array([1.0, 0, 0]), 0.0),
    "inverted": (np.array([1.0, 0, 0]), math.pi),
    "side+x": (np.array([0, 1.0, 0]), math.pi / 2),
    "side-x": (np.array([0, 1.0, 0]), -math.pi / 2),
    "side+y": (np.array([1.0, 0, 0]), math.pi / 2),
    "side-y": (np.array([1.0, 0, 0]), -math.pi / 2),
}


def orientations(n_random, rng):
    out = [(k, quat_from_axis_angle(a, t)) for k, (a, t) in AXIS_ORIENTATIONS.items()]
    for i in range(n_random):
        q = rng.normal(size=4)
        out.append((f"random{i}", q / np.linalg.norm(q)))
    return out


def _place(model, data, body, q, gap, pts):
    lowest = (pts @ rotmat(q).T)[:, 2].min()
    jid = model.body(body).jntadr[0]
    a = model.jnt_qposadr[jid]
    data.qpos[a:a + 3] = [0.0, 0.0, gap - lowest]
    data.qpos[a + 3:a + 7] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return a


def rest_class(q):
    z = rotmat(q)[:, 2]
    if z[2] > math.cos(math.radians(15)):
        return "upright"
    if z[2] < -math.cos(math.radians(15)):
        return "inverted"
    if abs(z[2]) < math.sin(math.radians(15)):
        return "side"
    return "tilted"


def drop_one(model, data, body, q, cfg, pts):
    s = cfg["sim"]
    mujoco.mj_resetData(model, data)
    a = _place(model, data, body, q, s["drop_height_m"], pts)
    m = total_mass(model, body)
    g = abs(model.opt.gravity[2])
    e_scale = m * g * s["drop_height_m"]
    e0 = data.energy[0] + data.energy[1]
    dt = model.opt.timestep
    n = int(round(s["drop_duration_s"] / dt))
    win = int(round(s["jitter_window_s"] / dt))
    lin = np.zeros(n)
    ang = np.zeros(n)
    emax = -np.inf
    pen_tail = 0.0
    nan = False
    for i in range(n):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            nan = True
            break
        v = data.qvel[0:3] if model.nv >= 6 else data.qvel[:3]
        w = data.qvel[3:6] if model.nv >= 6 else np.zeros(3)
        lin[i] = np.linalg.norm(v)
        ang[i] = np.linalg.norm(w)
        emax = max(emax, data.energy[0] + data.energy[1])
        if i >= n - win and data.ncon:
            pen_tail = max(pen_tail, -min(data.contact[k].dist for k in range(data.ncon)))
    warnings = int(sum(w.number for w in data.warning))
    still = (lin < s["settle_lin_mps"]) & (ang < s["settle_ang_rps"])
    settle_t = None
    if not nan and still[-1]:
        moving = np.where(~still)[0]
        settle_t = 0.0 if len(moving) == 0 else float((moving[-1] + 1) * dt)
    q_end = data.qpos[a + 3:a + 7].copy()
    return {
        "nan": nan,
        "warnings": warnings,
        "energy_gain_rel": float((emax - e0) / e_scale) if not nan else float("inf"),
        "settled": settle_t is not None,
        "settle_time_s": settle_t,
        "penetration_mm": float(pen_tail * 1e3),
        "jitter_rms_mmps": float(np.sqrt(np.mean(lin[-win:] ** 2)) * 1e3) if not nan else float("inf"),
        "rest": rest_class(q_end) if not nan else "nan",
        "final_tilt_deg": float(math.degrees(math.acos(np.clip(rotmat(q_end)[2, 2], -1, 1)))) if not nan else None,
        "final_speed_mmps": float(lin[-1] * 1e3) if not nan else None,
        "final_spin_rps": float(ang[-1]) if not nan else None,
    }


def drop_test(mjcf_path, cfg, rng, spec=None, timestep=None, which=None):
    model, data, body = free_scene(mjcf_path, spec=spec, timestep=timestep)
    pts = collision_points(model, body)
    runs = {}
    for name, q in orientations(cfg["sim"]["random_orientations"], rng):
        if which and name not in which:
            continue
        runs[name] = drop_one(model, data, body, q, cfg, pts)
    return runs


def drop_verdict(runs, cfg, with_warnings=False):
    s = cfg["sim"]
    problems = []
    warnings = []
    for k, r in runs.items():
        if r["nan"] or r["warnings"]:
            problems.append(f"{k}: unstable (nan={r['nan']}, warnings={r['warnings']})")
            continue
        if r["energy_gain_rel"] > s["energy_gain_rel"]:
            problems.append(f"{k}: energy gain {r['energy_gain_rel']:.1%} of drop energy")
        if r["penetration_mm"] > s["penetration_abs_m"] * 1e3:
            problems.append(f"{k}: penetration {r['penetration_mm']:.2f} mm in the final window")
    up = runs.get("upright")
    if up and not up["nan"]:
        if up["rest"] != "upright":
            problems.append(f"upright drop ended {up['rest']}")
        elif not up["settled"] or up["jitter_rms_mmps"] > s["jitter_lin_rms_mps"] * 1e3:
            warnings.append(f"upright drop did not come to rest (jitter {up['jitter_rms_mmps']:.2f} mm/s)")
    return (problems, warnings) if with_warnings else problems


def _slide_once(model, data, theta, s, pts, body):
    mujoco.mj_resetData(model, data)
    lowest = pts[:, 2].min()
    data.qpos[2] = 1e-4 - lowest
    model.opt.gravity[:] = [0, 0, -9.81]
    dt = model.opt.timestep
    for _ in range(int(0.2 / dt)):
        mujoco.mj_step(model, data)
    x0 = data.qpos[0]
    model.opt.gravity[:] = [9.81 * math.sin(theta), 0, -9.81 * math.cos(theta)]
    for _ in range(int(s["incline_hold_s"] / dt)):
        mujoco.mj_step(model, data)
    model.opt.gravity[:] = [0, 0, -9.81]
    return abs(data.qpos[0] - x0) > s["incline_slide_m"]


def critical_slide_angle(mjcf_path, cfg, spec=None):
    s = cfg["sim"]
    model, data, body = slide_scene(mjcf_path, spec=spec)
    pts = collision_points(model, body)
    lo, hi = 0.0, math.radians(s["incline_max_deg"])
    if not _slide_once(model, data, hi, s, pts, body):
        return None
    tol = math.radians(s["incline_tol_deg"])
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _slide_once(model, data, mid, s, pts, body):
            hi = mid
        else:
            lo = mid
    return math.degrees(0.5 * (lo + hi))


def asset_friction(model):
    pri = model.geom_priority
    mask = (model.geom_contype > 0) | (model.geom_conaffinity > 0)
    idx = [i for i in range(model.ngeom) if mask[i] and model.geom_bodyid[i] != 0]
    best = max(idx, key=lambda i: pri[i])
    return float(model.geom_friction[best, 0])


def roll_test(mjcf_path, cfg, spec=None):
    s = cfg["sim"]
    model, data, body = free_scene(mjcf_path, spec=spec, energy=False)
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    pts = collision_points(model, body)
    bid = model.body(body).id
    q = quat_from_axis_angle([1.0, 0, 0], math.pi / 2)
    _place(model, data, body, q, 1e-4, pts)
    dt = model.opt.timestep
    for _ in range(int(0.3 / dt)):
        mujoco.mj_step(model, data)
    theta = math.radians(s["roll_angle_deg"])
    model.opt.gravity[:] = [9.81 * math.sin(theta), 0, -9.81 * math.cos(theta)]
    for _ in range(int(0.05 / dt)):
        mujoco.mj_step(model, data)
    n = int(0.5 / dt)
    xs = np.zeros(n)
    for i in range(n):
        mujoco.mj_step(model, data)
        xs[i] = data.xipos[bid][0]
    model.opt.gravity[:] = [0, 0, -9.81]
    t = np.arange(1, n + 1) * dt
    a = 2 * np.polyfit(t, xs, 2)[0]
    r_iq = rotmat(model.body_iquat[bid])
    i_body = r_iq @ np.diag(model.body_inertia[bid]) @ r_iq.T
    com = model.body_ipos[bid]
    radius = float(np.sqrt(((pts[:, 0] - com[0]) ** 2 + (pts[:, 1] - com[1]) ** 2).max()))
    mass = float(model.body_mass[bid])
    expected = 1.0 / (1.0 + i_body[2, 2] / (mass * radius ** 2))
    return {
        "accel_ratio": float(a / (9.81 * math.sin(theta))),
        "expected_from_inertia": float(expected),
        "solid_cylinder": 2 / 3,
        "radius_mm": radius * 1e3,
        "k2_over_r2": float(i_body[2, 2] / (mass * radius ** 2)),
    }
