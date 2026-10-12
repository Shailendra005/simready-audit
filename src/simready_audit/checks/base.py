from dataclasses import dataclass, field

PASS, WARN, FAIL, INFO, SKIP = "PASS", "WARN", "FAIL", "INFO", "SKIP"
SEVERITY = {PASS: 0, INFO: 0, SKIP: 0, WARN: 1, FAIL: 2}

REGISTRY = []


@dataclass
class CheckResult:
    check_id: str
    title: str
    status: str
    metrics: dict = field(default_factory=dict)
    message: str = ""

    def to_dict(self):
        return {"check_id": self.check_id, "title": self.title, "status": self.status, "metrics": self.metrics, "message": self.message}


def check(check_id, title, group):
    def deco(fn):
        REGISTRY.append((check_id, title, group, fn))
        return fn
    return deco


def run_checks(ctx, groups=("contract", "geometry", "physical")):
    out = []
    for check_id, title, group, fn in REGISTRY:
        if group not in groups:
            continue
        try:
            status, metrics, message = fn(ctx)
        except Exception as e:
            status, metrics, message = FAIL, {}, f"check raised {type(e).__name__}: {e}"
        out.append(CheckResult(check_id, title, status, _clean(metrics), message))
    return out


def _clean(d):
    import numpy as np
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, np.ndarray):
            v = [float(x) for x in v.ravel()]
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        if isinstance(v, float):
            v = float(f"{v:.9g}")
        out[k] = v
    return out
