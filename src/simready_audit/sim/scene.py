import mujoco
import numpy as np


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotmat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def load_spec(mjcf_path=None, xml=None):
    if xml is not None:
        return mujoco.MjSpec.from_string(xml)
    return mujoco.MjSpec.from_file(str(mjcf_path))


def strip_visuals(spec):
    for g in list(spec.geoms):
        if g.contype == 0 and g.conaffinity == 0:
            spec.delete(g)
    for t in list(spec.textures):
        spec.delete(t)
    for mat in list(spec.materials):
        spec.delete(mat)
    used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    for m in list(spec.meshes):
        if m.name not in used:
            spec.delete(m)
    return spec


def add_floor(spec, friction=1.0, priority=0):
    g = spec.worldbody.add_geom()
    g.name = "floor"
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [5, 5, 0.1]
    g.friction = [friction, 0.005, 0.0001]
    g.priority = priority
    return g


def root_body(spec):
    bodies = [b for b in spec.bodies if b.name and b.name != "world"]
    tops = [b for b in bodies if b.parent.name == "world"] if bodies and hasattr(bodies[0], "parent") else bodies[:1]
    return tops[0] if tops else bodies[0]


def free_scene(mjcf_path=None, spec=None, cone=None, timestep=None, energy=True):
    spec = spec or load_spec(mjcf_path)
    strip_visuals(spec)
    body = root_body(spec)
    if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints):
        body.add_freejoint()
    add_floor(spec)
    if cone is not None:
        spec.option.cone = cone
    if timestep is not None:
        spec.option.timestep = timestep
    if energy:
        spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    model = spec.compile()
    return model, mujoco.MjData(model), body.name


def slide_scene(mjcf_path=None, spec=None, cone=mujoco.mjtCone.mjCONE_ELLIPTIC):
    spec = spec or load_spec(mjcf_path)
    strip_visuals(spec)
    body = root_body(spec)
    for axis, name in (([1, 0, 0], "sx"), ([0, 1, 0], "sy"), ([0, 0, 1], "sz")):
        j = body.add_joint()
        j.name = name
        j.type = mujoco.mjtJoint.mjJNT_SLIDE
        j.axis = axis
    add_floor(spec)
    spec.option.cone = cone
    model = spec.compile()
    return model, mujoco.MjData(model), body.name


def collision_points(model, body_name):
    bid = model.body(body_name).id
    pts = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid or (model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0):
            continue
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            v = model.mesh_vert[a:a + n]
            r = rotmat(model.geom_quat[gid])
            pts.append(v @ r.T + model.geom_pos[gid])
        else:
            s = model.geom_rbound[gid]
            c = model.geom_pos[gid]
            pts.append(c + s * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]))
    return np.vstack(pts)


def total_mass(model, body_name):
    return float(model.body_subtreemass[model.body(body_name).id])
