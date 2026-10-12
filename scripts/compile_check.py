import glob
import json
import sys

import mujoco

paths = sorted(glob.glob(sys.argv[1] + "/assets/*/*/mjcf/*.xml"))
errors = {}
for p in paths:
    try:
        mujoco.MjModel.from_xml_path(p)
    except Exception as e:
        errors[p] = str(e)
print(json.dumps({"mujoco": mujoco.__version__, "files": len(paths), "errors": len(errors), "examples": list(errors.items())[:3]}))
