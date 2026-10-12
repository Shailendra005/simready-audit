import numpy as np

from ..analytics import mesh as M
from .base import FAIL, INFO, PASS, SKIP, WARN, check
from .contract import expected_capacity_ml


def _bbox_mm(mesh):
    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
    return (hi - lo) * 1000


@check("C1", "Dimensions match the dataset card", "physical")
def dimensions(ctx):
    d = ctx.cfg["dimensions"]
    cls = ctx.usd.cls
    if cls == "beverage_can":
        ext = _bbox_mm(ctx.collider)
        dia, h = float(max(ext[:2])), float(ext[2])
        slim = d["can_slim_diameter_mm"][0] - d["can_diameter_tol_mm"] <= dia <= d["can_slim_diameter_mm"][1] + d["can_diameter_tol_mm"]
        std = abs(dia - d["can_diameter_mm"][0]) <= d["can_diameter_tol_mm"]
        hmatch = min(abs(h - x) for x in d["can_heights_mm"]) <= d["can_height_tol_mm"]
        m = {"diameter_mm": dia, "height_mm": h, "slim": bool(slim and not std)}
        if std and hmatch:
            return PASS, m, ""
        if slim:
            return INFO, m, "slim/sleek can (card lists no dimensions)"
        return WARN, m, "diameter or height matches no listed can format"
    if cls == "mug":
        z = ctx.body_zone
        if z is None:
            return SKIP, {}, "no body zone"
        ext = _bbox_mm(z)
        dia, h = float(max(ext[:2])), float(ext[2])
        t = d["mug_tol_mm"]
        okd = d["mug_body_diameter_mm"][0] - t <= dia <= d["mug_body_diameter_mm"][1] + t
        okh = d["mug_body_height_mm"][0] - t <= h <= d["mug_body_height_mm"][1] + t
        m = {"body_diameter_mm": dia, "body_height_mm": h}
        return (PASS if okd and okh else WARN), m, "" if okd and okh else "mug body outside the card's stated diameter/height range"
    if cls == "glass_jar":
        cap, _ = expected_capacity_ml(ctx)
        best = min(d["jar_capacities_ml"], key=lambda x: abs(x - cap))
        rel = abs(cap - best) / best
        m = {"interior_ml": cap, "nearest_format_ml": best, "rel": rel}
        return (PASS if rel <= d["jar_capacity_rel_tol"] else WARN), m, "" if rel <= d["jar_capacity_rel_tol"] else "interior volume matches no listed jar format"
    ext = _bbox_mm(ctx.collider)
    return INFO, {"extent_mm": ext}, "no dimension spec for class"


def density_ratio(ctx):
    mat = ctx.material
    if mat is None or mat.density is None:
        return None, None
    ru = ctx.usd.mass / (mat.density * ctx.solid["volume"])
    vols = [sum(M.mass_properties(p.vertices, p.faces)["volume"] for p in ctx.collider_parts)]
    if "solid_visual" in ctx.mass_models:
        vols.append(ctx.mass_models["solid_visual"]["volume"])
    rs = min((ctx.usd.mass / (mat.density * v) for v in vols), key=lambda r: abs(r - 1))
    return ru, rs


@check("C2", "Mass equals density x collider volume", "physical")
def mass_volume(ctx):
    ru, rs = density_ratio(ctx)
    if ru is None:
        return SKIP, {}, "no authored density"
    c = ctx.cfg["mass"]
    dev = min(abs(ru - 1), abs(rs - 1))
    m = {"ratio_union": ru, "ratio_best_other_volume": rs, "implied_density": ctx.material.density * ru, "authored_density": ctx.material.density}
    if ctx.usd.cls in c["thin_wall_classes"]:
        m["implied_uniform_wall_mm"] = ctx.usd.mass / (ctx.material.density * ctx.shell["area"]) * 1e3
        return INFO, m, f"thin-walled: mass is not density x collider volume ({ru:.3g}x); treated as a shell of {m['implied_uniform_wall_mm']:.3f} mm"
    if dev <= c["density_ratio_pass"]:
        return PASS, m, ""
    if dev <= c["density_ratio_warn"]:
        return WARN, m, f"mass is {ru:.2f}x density x collider volume"
    return FAIL, m, f"mass is {ru:.3g}x density x collider volume"


ALUMINIUM_MIN_WALL_M = 0.097e-3


@check("C3", "Mass plausible against cited real-world values", "physical")
def real_world(ctx):
    mat = ctx.material
    if ctx.usd.cls != "beverage_can" or mat is None or mat.density is None:
        return SKIP, {}, "no cited reference for class"
    t = ctx.usd.mass / (mat.density * ctx.shell["area"])
    m = {"mass_g": ctx.usd.mass * 1e3, "implied_wall_mm": t * 1e3, "reference_min_wall_mm": ALUMINIUM_MIN_WALL_M * 1e3, "surface_cm2": ctx.shell["area"] * 1e4}
    if t < ALUMINIUM_MIN_WALL_M:
        return WARN, m, f"implied uniform wall {t*1e3:.3f} mm is thinner than the thinnest commercial can wall (0.097 mm)"
    return PASS, m, ""


@check("C4", "Inertia is physically valid", "physical")
def inertia_valid(ctx):
    i = ctx.usd.inertia_tensor
    if i is None:
        return SKIP, {}, ""
    ev = np.sort(np.linalg.eigvalsh(0.5 * (i + i.T)))
    slack = ctx.cfg["inertia"]["triangle_rel_slack"] * ev[-1]
    ok = ev[0] > 0 and ev[0] + ev[1] + slack >= ev[2]
    return (PASS if ok else FAIL), {"principal": ev}, "" if ok else "inertia not positive definite or violates triangle inequality"


def mass_model(ctx):
    thin = ctx.usd.cls in ctx.cfg["mass"]["thin_wall_classes"]
    return ("shell" if thin else "solid"), (ctx.shell if thin else ctx.solid)


def _project(t, axes):
    return axes.T @ (0.5 * (t + t.T)) @ axes


@check("C5", "Inertia consistent with the mass distribution", "physical")
def inertia_consistency(ctx):
    i = ctx.usd.inertia_tensor
    if i is None:
        return SKIP, {}, "inertia not authored"
    tol = ctx.cfg["inertia"]["k2_rel_fail"]
    k2 = i / ctx.usd.mass
    ks = ctx.solid["inertia"] / ctx.solid["mass"]
    kh = ctx.shell["inertia"] / ctx.shell["mass"]
    rs = float(np.linalg.norm(k2 - ks) / np.linalg.norm(ks))
    rh = float(np.linalg.norm(k2 - kh) / np.linalg.norm(kh))
    _, axes = np.linalg.eigh(0.5 * (ks + ks.T))
    pa = _project(k2, axes)
    e = np.diag(pa)
    cands = np.array([np.diag(_project(mm["inertia"] / mm["mass"], axes)) for mm in ctx.mass_models.values()])
    lo = cands.min(0) * (1 - tol)
    hi = cands.max(0) * (1 + tol)
    off = float(np.abs(pa - np.diag(e)).max() / max(e.max(), 1e-30))
    in_band = bool(np.all((e >= lo) & (e <= hi)) and off <= tol)
    thin = ctx.usd.cls in ctx.cfg["mass"]["thin_wall_classes"]
    ph = np.diag(_project(kh, axes))
    m = {"rel_vs_solid": rs, "rel_vs_shell": rh, "closest": "solid" if rs < rh else "shell", "within_solid_shell_band": in_band, "off_diagonal_rel": off, "declared_thin_walled": thin, "min_axis_ratio_vs_shell": float(np.min(e / ph))}
    if thin and rh > tol:
        return FAIL, m, f"card describes a thin-walled object, but inertia per unit mass matches a {m['closest']} body ({rh:.0%} from the shell model)"
    if not thin and not in_band:
        return FAIL, m, "inertia per unit mass lies outside the range any mass distribution inside this shape can produce"
    return PASS, m, ""


@check("C6", "Centre of mass consistent with geometry", "physical")
def com_consistency(ctx):
    if ctx.usd.com is None:
        return SKIP, {}, ""
    lo, hi = ctx.collider.vertices.min(0), ctx.collider.vertices.max(0)
    diag = float(np.linalg.norm(hi - lo))
    cents = np.array([mm["com"] for mm in ctx.mass_models.values()])
    box_lo, box_hi = cents.min(0), cents.max(0)
    gap = np.maximum(0, np.maximum(box_lo - ctx.usd.com, ctx.usd.com - box_hi))
    d = float(np.linalg.norm(gap))
    rel = d / diag
    inside_hull = bool(np.all(ctx.usd.com >= lo - 1e-9) and np.all(ctx.usd.com <= hi + 1e-9))
    m = {"distance_to_model_range_mm": d * 1e3, "rel_bbox_diag": rel, "inside_bbox": inside_hull}
    if not inside_hull:
        return FAIL, m, "COM outside the object's bounding box"
    if rel > ctx.cfg["com"]["rel_bbox_diag_warn"]:
        return WARN, m, f"COM {d*1e3:.1f} mm outside the range spanned by solid and shell centroids"
    return PASS, m, ""
