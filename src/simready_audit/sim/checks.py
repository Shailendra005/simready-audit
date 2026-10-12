import math

import numpy as np

from ..checks.base import FAIL, INFO, PASS, SKIP, WARN, CheckResult, _clean
from .cavity import cavity_probe
from .tests import critical_slide_angle, drop_test, drop_verdict, roll_test

ROUND_CLASSES = {"beverage_can", "glass_jar"}


def _r(cid, title, status, metrics, msg=""):
    return CheckResult(cid, title, status, _clean(metrics), msg).to_dict()


def sim_checks(ctx, mjcf_path, spec_factory=None, include=("D1", "D2", "D3", "D4", "D5")):
    cfg = ctx.cfg
    s = cfg["sim"]
    out = []
    spec = (lambda: spec_factory()) if spec_factory else (lambda: None)
    if "D1" in include:
        runs = drop_test(mjcf_path, cfg, np.random.default_rng(ctx.seed), spec=spec())
        problems, warns = drop_verdict(runs, cfg, with_warnings=True)
        rests = {k: r["rest"] for k, r in runs.items()}
        worst_pen = max(r["penetration_mm"] for r in runs.values())
        metrics = {
            "orientations": len(runs),
            "max_penetration_mm": worst_pen,
            "max_energy_gain_rel": max(r["energy_gain_rel"] for r in runs.values()),
            "upright_settle_s": runs["upright"]["settle_time_s"],
            "rest_upright": sum(v == "upright" for v in rests.values()),
            "rest_side": sum(v == "side" for v in rests.values()),
            "rest_inverted": sum(v == "inverted" for v in rests.values()),
            "rest_tilted": sum(v == "tilted" for v in rests.values()),
        }
        status = FAIL if problems else (WARN if warns else PASS)
        out.append(_r("D1", "Drop test: stable, no energy gain, no sinking", status, metrics, "; ".join(problems + warns)))
        base = {k: r["rest"] for k, r in runs.items()}
    if "D2" in include:
        changes, unstable = [], []
        which = ["upright", "side+x", "random0"]
        for dt in s["timesteps"]:
            r2 = drop_test(mjcf_path, cfg, np.random.default_rng(ctx.seed), spec=spec(), timestep=dt, which=which)
            for k, r in r2.items():
                if r["nan"] or r["warnings"] or r["energy_gain_rel"] > s["energy_gain_rel"]:
                    unstable.append(f"dt={dt} {k}: unstable")
                elif r["penetration_mm"] > s["penetration_abs_m"] * 1e3:
                    changes.append(f"dt={dt} {k}: penetration {r['penetration_mm']:.2f} mm")
            if "D1" in include and r2["upright"]["rest"] != base.get("upright"):
                changes.append(f"dt={dt} upright: {base.get('upright')}->{r2['upright']['rest']}")
        status = FAIL if unstable else (WARN if changes else PASS)
        out.append(_r("D2", "Timestep sensitivity (0.5/1/2 ms)", status, {"rest_changes": len(changes), "unstable": len(unstable)}, "; ".join(unstable + changes)))
    if "D3" in include:
        mat = ctx.material
        theta = critical_slide_angle(mjcf_path, cfg, spec=spec())
        if mat is None:
            out.append(_r("D3", "Slide angle matches authored friction", SKIP, {}, "no USD material"))
        elif theta is None:
            target = math.degrees(math.atan(mat.static_friction))
            status = FAIL if target + s["friction_angle_fail_deg"] < s["incline_max_deg"] else PASS
            out.append(_r("D3", "Slide angle matches authored friction", status, {"slide_angle_deg": None, "usd_static_angle_deg": target}, f"does not slide up to {s['incline_max_deg']} deg; USD implies {target:.1f} deg" if status == FAIL else ""))
        else:
            target = math.degrees(math.atan(mat.static_friction))
            lo = math.degrees(math.atan(mat.dynamic_friction))
            err = 0.0 if lo - s["friction_angle_fail_deg"] <= theta <= target + s["friction_angle_fail_deg"] else min(abs(theta - target), abs(theta - lo))
            status = FAIL if err > 0 else PASS
            out.append(_r("D3", "Slide angle matches authored friction", status, {"slide_angle_deg": theta, "usd_static_angle_deg": target, "usd_dynamic_angle_deg": lo}, f"slides at {theta:.1f} deg; USD friction implies {lo:.1f}-{target:.1f} deg" if status == FAIL else ""))
    if "D4" in include:
        if ctx.usd.cls == "beverage_can":
            r = roll_test(mjcf_path, cfg, spec=spec())
            out.append(_r("D4", "Can rolling acceleration", INFO, r, f"rolls at {r['accel_ratio']:.3f} g sin(theta); inertia implies {r['expected_from_inertia']:.3f}"))
        else:
            out.append(_r("D4", "Can rolling acceleration", SKIP, {}, "not a can"))
    if "D5" in include:
        if ctx.usd.cls == "mug" and ctx.body_zone is not None:
            r = cavity_probe(mjcf_path, ctx.body_zone, cfg, spec=spec())
            status = PASS if r["result"] == "ENTERED" else (FAIL if r["result"] == "BLOCKED" else WARN)
            out.append(_r("D5", "Cavity probe: a ball dropped into the cup enters it", status, r, "ball rests on an invisible lid at the rim" if r["result"] == "BLOCKED" else ""))
        else:
            out.append(_r("D5", "Cavity probe: a ball dropped into the cup enters it", SKIP, {}, "not an open container"))
    return out

