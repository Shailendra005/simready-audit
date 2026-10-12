import json
import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "bottles-and-containers"


def _raw():
    if not (RES / "assets").exists():
        pytest.skip("no results yet")
    return [json.loads(p.read_text()) for p in sorted((RES / "assets").glob("*.json"))]


def _count(assets, cid, status, cls=None):
    return sum(1 for a in assets if (cls is None or a["cls"] == cls) for c in a["checks"] if c["check_id"] == cid and c["status"] == status)


def test_readme_results_table_matches_raw_results():
    assets = _raw()
    readme = (ROOT / "README.md").read_text()
    n = Counter(a["cls"] for a in assets)
    for cid in ("A7", "A8", "A10", "B1", "C5", "D3", "D5"):
        cells = []
        for cls in ("beverage_can", "glass_jar", "mug"):
            p, w, f = (_count(assets, cid, s, cls) for s in ("PASS", "WARN", "FAIL"))
            cells.append("–" if p + w + f == 0 else f"{p} / {w} / {f}")
        rows = [r for r in re.findall(rf"^\| {cid} [^|]*\|.*$", readme, re.M) if " / " in r or "–" in r.split("|")[2]]
        assert len(rows) == 1, rows
        assert rows[0].split("|")[2:5] == [f" {c} " for c in cells], (cid, rows[0], cells)
    assert f"Cans ({n['beverage_can']})" in readme and f"Mugs ({n['mug']})" in readme


def test_readme_mutation_table_matches_raw_results():
    p = RES / "mutation.json"
    if not p.exists():
        pytest.skip("no results yet")
    m = json.loads(p.read_text())
    readme = (ROOT / "README.md").read_text()
    for r in m["detection"]:
        line = next(l for l in readme.splitlines() if l.startswith(f"| {r['mutation']} |"))
        expected = f"{r['detected']}/{r['eligible']}" if r["eligible"] else "covered by a unit test)"
        assert line.rstrip(" |").endswith(expected), (line, expected)
    changes = sum(x["verdict_changes"] for x in m["invariance"])
    comparisons = sum(x["verdict_comparisons"] for x in m["invariance"])
    assert f"Across {comparisons} pass/warn/fail verdicts (information-only checks excluded), {changes} changed." in readme
