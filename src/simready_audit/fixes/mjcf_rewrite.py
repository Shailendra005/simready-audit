import xml.etree.ElementTree as ET
from pathlib import Path



def _abs_assets(root, base):
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texdir = compiler.get("texturedir", "") if compiler is not None else ""
    for m in root.iter("mesh"):
        if m.get("file"):
            m.set("file", str((base / meshdir / m.get("file")).resolve()))
    for t in root.iter("texture"):
        if t.get("file"):
            t.set("file", str((base / texdir / t.get("file")).resolve()))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)


def write_obj(path, vertices, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for t in faces:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def rewrite(src_xml, out_xml, hulls=None, friction=None, solref=None):
    src_xml, out_xml = Path(src_xml), Path(out_xml)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(src_xml)
    root = tree.getroot()
    _abs_assets(root, src_xml.parent)
    if friction is not None or solref is not None:
        for d in root.iter("default"):
            if d.get("class") == "collision":
                g = d.find("geom")
                if friction is not None:
                    vals = g.get("friction", "1 0.005 0.0001").split()
                    vals[0] = f"{friction:.6g}"
                    g.set("friction", " ".join(vals))
                if solref is not None:
                    g.set("solref", " ".join(f"{x:.6g}" for x in solref))
    if hulls is not None:
        asset = root.find("asset")
        body = root.find("worldbody").find("body")
        for g in list(body.findall("geom")):
            if g.get("class") == "collision":
                body.remove(g)
        coll_names = {g.get("mesh") for g in root.iter("geom") if g.get("class") == "collision"}
        for m in list(asset.findall("mesh")):
            if m.get("name", "").startswith("col_") and m.get("name") not in coll_names:
                asset.remove(m)
        hull_dir = out_xml.parent / "collision"
        hull_dir.mkdir(parents=True, exist_ok=True)
        for i, (v, f) in enumerate(hulls):
            name = f"col_fix_{i:03d}"
            write_obj(hull_dir / f"{name}.obj", v, f)
            m = ET.SubElement(asset, "mesh")
            m.set("name", name)
            m.set("file", str((hull_dir / f"{name}.obj").resolve()))
            g = ET.SubElement(body, "geom")
            g.set("class", "collision")
            g.set("type", "mesh")
            g.set("mesh", name)
    ET.indent(tree)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    return out_xml
