import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from ..sim.cavity import opening_geometry
from ..sim.scene import add_floor, load_spec

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e7e6e2"
SURFACE = "#fcfcfb"
S1 = "#2a78d6"
S2 = "#eb6834"
S3 = "#1baf7a"


def _rgba(hexcolor, alpha):
    h = hexcolor.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [alpha]


def probe_scene(mjcf_path, body_zone, cfg, collision_rgba, visual_alpha=0.28, duration=1.5):
    s = cfg["sim"]
    geo = opening_geometry(body_zone)
    r = float(np.clip(s["probe_radius_frac"] * geo["r_inner"], s["probe_radius_min_m"], s["probe_radius_max_m"]))
    spec = load_spec(mjcf_path)
    model0 = load_spec(mjcf_path).compile()
    hull_meshes = []
    for gid in range(model0.ngeom):
        if model0.geom_contype[gid] == 0 and model0.geom_conaffinity[gid] == 0:
            continue
        if model0.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model0.geom_dataid[gid]
        a, n = model0.mesh_vertadr[mid], model0.mesh_vertnum[mid]
        v = model0.mesh_vert[a:a + n]
        from ..sim.scene import rotmat
        v = v @ rotmat(model0.geom_quat[gid]).T + model0.geom_pos[gid]
        from ..analytics.mesh import convex_hull
        from ..io.types import TriMesh
        h = convex_hull(TriMesh("h", v, np.zeros((0, 3), np.int64)))
        hull_meshes.append((h.vertices, h.faces))
    body = None
    for g in spec.geoms:
        if g.contype == 0 and g.conaffinity == 0:
            g.material = ""
            g.rgba = [0.55, 0.55, 0.53, visual_alpha]
            body = g.parent
        else:
            g.rgba = [0, 0, 0, 0]
    for k, (v, f) in enumerate(hull_meshes):
        mesh = spec.add_mesh()
        mesh.name = f"hullviz_{k}"
        mesh.uservert = v.ravel().tolist()
        mesh.userface = f.ravel().tolist()
        g = body.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = mesh.name
        g.contype = 0
        g.conaffinity = 0
        g.rgba = collision_rgba
        g.group = 1
    for mat in list(spec.materials):
        spec.delete(mat)
    for t in list(spec.textures):
        spec.delete(t)
    sky = spec.add_texture()
    sky.name = "sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    sky.rgb1 = [0.988, 0.988, 0.984]
    sky.rgb2 = [0.988, 0.988, 0.984]
    sky.width = 64
    sky.height = 64
    floor = add_floor(spec)
    floor.rgba = [0.988, 0.988, 0.984, 1]
    probe = spec.worldbody.add_body()
    probe.name = "probe"
    probe.pos = [geo["center"][0], geo["center"][1], geo["z_rim"] + s["probe_drop_above_rim_m"] + r]
    probe.add_freejoint()
    b = probe.add_geom()
    b.type = mujoco.mjtGeom.mjGEOM_SPHERE
    b.size = [r, 0, 0]
    b.mass = 0.005
    b.rgba = _rgba(S1, 1.0)
    light = spec.worldbody.add_light()
    light.pos = [0.3, -0.3, 0.8]
    light.dir = [-0.3, 0.3, -0.8]
    light.diffuse = [0.7, 0.7, 0.7]
    spec.visual.headlight.ambient = [0.6, 0.6, 0.6]
    spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    spec.visual.global_.offwidth = 1024
    spec.visual.global_.offheight = 1024
    model = spec.compile()
    data = mujoco.MjData(model)
    for _ in range(int(duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return model, data, geo


def render(model, data, lookat, distance, azimuth=135, elevation=-28, size=(720, 720)):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    opt = mujoco.MjvOption()
    for gi in range(6):
        opt.geomgroup[gi] = 1 if gi in (0, 1, 2) else 0
    with mujoco.Renderer(model, size[1], size[0]) as r:
        r.update_scene(data, camera=cam, scene_option=opt)
        return r.render()


def probe_pair_figure(left, right, out_png, title_left, title_right):
    import textwrap
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.8), facecolor=SURFACE, gridspec_kw={"wspace": 0.06})
    for ax, (img, sub), t in zip(axes, (left, right), (title_left, title_right)):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(t, fontsize=12.5, color=INK, loc="left", pad=6, fontweight="bold")
        ax.text(0.0, -0.03, "\n".join(textwrap.wrap(sub, 52)), transform=ax.transAxes, fontsize=10, color=INK2, va="top", ha="left", linespacing=1.4)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.2)
    fig.savefig(out_png, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return out_png
