import numpy as np

from .base import FAIL, INFO, PASS, SKIP, WARN, check


@check("A1", "Stage units and up axis", "contract")
def units(ctx):
    c = ctx.cfg["units"]
    u = ctx.usd
    ok = abs(u.meters_per_unit - c["meters_per_unit"]) < 1e-12 and abs(u.kilograms_per_unit - c["kilograms_per_unit"]) < 1e-12 and u.up_axis == c["up_axis"]
    m = {"meters_per_unit": u.meters_per_unit, "kilograms_per_unit": u.kilograms_per_unit, "up_axis": u.up_axis}
    return (PASS if ok else FAIL), m, "" if ok else "stage units or up axis differ from metres / kilograms / Z-up"


@check("A2", "Physics schema present and valid", "contract")
def schema(ctx):
    u = ctx.usd
    problems = []
    if "PhysicsRigidBodyAPI" not in u.applied_schemas:
        problems.append("root lacks PhysicsRigidBodyAPI")
    densities = [c.material.density for c in u.colliders if c.material is not None and c.material.density]
    if (u.mass is None or not u.mass > 0) and not densities:
        problems.append("no mass and no density: engine default density applies")
    notes = []
    if u.inertia_tensor is None:
        notes.append("inertia not authored (engine derives it from colliders)")
    if not u.colliders:
        problems.append("no collider")
    for c in u.colliders:
        if c.approximation is None:
            problems.append(f"{c.name}: no approximation")
        if c.material is None:
            problems.append(f"{c.name}: no bound physics material: engine default friction/restitution apply")
            continue
        mat = c.material
        if mat.static_friction is None or mat.dynamic_friction is None or mat.static_friction < mat.dynamic_friction or mat.dynamic_friction < 0:
            problems.append(f"{c.name}: friction pair invalid")
        if mat.restitution is None or not 0 <= mat.restitution <= 1:
            problems.append(f"{c.name}: restitution outside [0, 1]")
        if mat.density is not None and not mat.density > 0:
            problems.append(f"{c.name}: density not positive")
    problems = sorted(set(problems))
    return (FAIL if problems else PASS), {"problems": len(problems), "inertia_authored": u.inertia_tensor is not None}, "; ".join(problems + notes)


@check("A3", "MJCF compiles in MuJoCo", "contract")
def compiles(ctx):
    if ctx.mjcf is None:
        return SKIP, {}, "no MJCF"
    err = ctx.mjcf.compile_error
    return (FAIL if err else PASS), {"has_freejoint": ctx.mjcf.has_freejoint, "bodies": len(ctx.mjcf.bodies)}, err or ""


def _single_body(ctx):
    if ctx.mjcf is None or ctx.mj_body is None:
        return None
    if len(ctx.mjcf.bodies) != 1:
        return None
    return ctx.mj_body


@check("A4", "Mass parity USD vs MJCF", "contract")
def mass_parity(ctx):
    b = _single_body(ctx)
    if b is None or b.mass is None:
        return SKIP, {}, "no single MJCF body with explicit mass"
    rel = abs(b.mass - ctx.usd.mass) / ctx.usd.mass
    return (PASS if rel <= ctx.cfg["parity"]["mass_rel"] else FAIL), {"usd_kg": ctx.usd.mass, "mjcf_kg": b.mass, "rel": rel}, ""


@check("A5", "Centre-of-mass parity USD vs MJCF", "contract")
def com_parity(ctx):
    b = _single_body(ctx)
    if b is None or b.com is None or ctx.usd.com is None:
        return SKIP, {}, "no single MJCF body with explicit COM"
    d = float(np.linalg.norm(b.com - ctx.usd.com))
    return (PASS if d <= ctx.cfg["parity"]["com_abs_m"] else FAIL), {"distance_m": d}, ""


@check("A6", "Inertia tensor parity USD vs MJCF", "contract")
def inertia_parity(ctx):
    b = _single_body(ctx)
    if b is None or b.inertia_tensor is None or ctx.usd.inertia_tensor is None:
        return SKIP, {}, "no single MJCF body with explicit inertia"
    a, m = ctx.usd.inertia_tensor, b.inertia_tensor
    rel = float(np.linalg.norm(a - m) / np.linalg.norm(a))
    return (PASS if rel <= ctx.cfg["parity"]["inertia_rel_frobenius"] else FAIL), {"rel_frobenius": rel}, ""


@check("A7", "MJCF friction within USD static/dynamic interval", "contract")
def friction(ctx):
    b = ctx.mj_body
    mat = ctx.material
    if b is None or mat is None or not b.collision_geoms:
        return SKIP, {}, "no MJCF collision geoms or USD material"
    mus = sorted({float(g.friction[0]) if g.friction is not None else 1.0 for g in b.collision_geoms})
    slack = ctx.cfg["friction"]["interval_slack"]
    lo, hi = mat.dynamic_friction - slack, mat.static_friction + slack
    bad = [m for m in mus if not lo <= m <= hi]
    metrics = {"mjcf_mu": mus[0] if len(mus) == 1 else mus, "usd_static": mat.static_friction, "usd_dynamic": mat.dynamic_friction}
    return (FAIL if bad else PASS), metrics, f"MJCF sliding friction {bad} outside USD [{mat.dynamic_friction:.3g}, {mat.static_friction:.3g}]" if bad else ""


@check("A8", "Restitution carried into MJCF contact", "contract")
def restitution(ctx):
    b = ctx.mj_body
    mat = ctx.material
    if b is None or mat is None or not b.collision_geoms:
        return SKIP, {}, ""
    solref = b.collision_geoms[0].solref
    if solref is None:
        solref = np.array([0.02, 1.0])
    damp = float(solref[1]) if len(solref) > 1 else 1.0
    e = float(np.float32(mat.restitution or 0.0))
    lost = e > ctx.cfg["restitution"]["warn_if_usd_above"] + 1e-6 and solref[0] > 0 and damp >= 1.0
    return (WARN if lost else PASS), {"usd_restitution": e, "mjcf_solref": solref, "mjcf_dampratio": damp}, "USD restitution > 0 but MJCF contact is critically damped (no bounce)" if lost else ""


@check("A9", "Collider approximation intent vs MJCF geoms", "contract")
def approximation_intent(ctx):
    b = ctx.mj_body
    if b is None:
        return SKIP, {}, ""
    approx = sorted({c.approximation for c in ctx.usd.colliders})
    n = len([g for g in b.collision_geoms if g.kind == "mesh"])
    nonconvex = any(a in ("sdf", "convexDecomposition", "none", None) for a in approx)
    msg = f"USD asks for {approx}; MJCF has {n} mesh geom(s), each convex-hulled by MuJoCo" if nonconvex and n == 1 else ""
    return INFO, {"usd_approximation": approx, "mjcf_mesh_geoms": n}, msg


CAN_FORMATS_ML = {115: 330, 122: 355, 149: 440, 168: 500, 189: 568}


def expected_capacity_ml(ctx):
    cls = ctx.usd.cls
    if cls == "beverage_can":
        lo, hi = ctx.collider.vertices.min(0), ctx.collider.vertices.max(0)
        h = (hi - lo)[2] * 1000
        d = max((hi - lo)[:2]) * 1000
        if abs(d - 66.0) > ctx.cfg["dimensions"]["can_diameter_tol_mm"]:
            return None, "slim or sleek can: no format table"
        best = min(CAN_FORMATS_ML, key=lambda k: abs(k - h))
        if abs(best - h) > ctx.cfg["dimensions"]["can_height_tol_mm"]:
            return None, "height matches no listed format"
        return float(CAN_FORMATS_ML[best]), f"card format for {best} mm can"
    if cls == "mug" and ctx.body_zone is not None:
        from ..analytics import mesh as M
        z = ctx.body_zone
        h = M.convex_hull(z)
        return (M.mass_properties(h.vertices, h.faces)["volume"] - M.mass_properties(z.vertices, z.faces)["volume"]) * 1e6, "hull(body zone) minus body material"
    if cls == "glass_jar":
        return (ctx.envelope_volume - ctx.solid["volume"]) * 1e6, "envelope minus collider material"
    return None, "no cavity model for class"


@check("A10", "max_mass_kg matches documented meaning (cavity x water)", "contract")
def max_mass(ctx):
    kd = ctx.usd.custom.get("kaedim", {})
    if "max_mass_kg" not in kd:
        return SKIP, {}, "no max_mass_kg"
    c = ctx.cfg["max_mass"]
    mm = float(kd["max_mass_kg"])
    m = ctx.usd.mass
    claim_ml = mm / c["water_density"] * 1e6
    envelope_ml = ctx.envelope_volume * 1e6
    expected_ml, basis = expected_capacity_ml(ctx)
    equals = abs(mm - m) <= max(c["equals_mass_rel"] * m, c["equals_mass_abs_kg"])
    exceeds = claim_ml > envelope_ml
    metrics = {"max_mass_kg": mm, "mass_kg": m, "claimed_capacity_ml": claim_ml, "envelope_ml": envelope_ml, "expected_capacity_ml": expected_ml, "basis": basis}
    if equals:
        return FAIL, metrics, "max_mass_kg equals the physics mass, not cavity volume x water density"
    if exceeds:
        return FAIL, metrics, "claimed capacity exceeds the object's entire convex envelope"
    if expected_ml:
        rel = abs(claim_ml - expected_ml) / expected_ml
        metrics["rel_vs_expected"] = rel
        if rel > c["cavity_rel_warn"]:
            return WARN, metrics, f"claimed capacity differs from {basis} by {rel:.0%}"
    return PASS, metrics, ""
