import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import yaml

from simready_audit.io.mjcf_loader import read_obj
from simready_audit.io.types import TriMesh
from simready_audit.sim.cavity import cavity_probe
from simready_audit.sim.tests import critical_slide_angle

cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "thresholds.v1.yaml").read_text())
out = []
for vid in sys.argv[1:]:
    base = Path("data/bottles-and-containers/assets/mug") / vid / "mjcf"
    v, f = read_obj(base / "visuals" / "vis_base__body.obj")
    probe = cavity_probe(base / f"{vid}.xml", TriMesh("vis_base__body", v, f), cfg)
    out.append({"vid": vid, "probe": probe["result"], "slide_deg": round(critical_slide_angle(base / f"{vid}.xml", cfg), 2)})
print(json.dumps({"mujoco": mujoco.__version__, "results": out}))
