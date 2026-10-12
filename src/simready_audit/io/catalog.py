from pathlib import Path


def iter_variants(dataset_root):
    root = Path(dataset_root) / "assets"
    for usda in sorted(root.glob("*/*/*.usda")):
        vid = usda.stem
        mjcf = usda.parent / "mjcf" / f"{vid}.xml"
        yield {"vid": vid, "cls": usda.parent.parent.name, "usda": usda, "mjcf": mjcf if mjcf.exists() else None}
