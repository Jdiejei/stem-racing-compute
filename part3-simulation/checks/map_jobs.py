"""Queue jobs for one start car's compass map (run on Oracle or in Actions, never on the PC).

    python map_jobs.py base  <start.json> <ref> > batch.json      # the start car, saves its case
    python map_jobs.py modes <start.json> <ref> <case_url> <sha256> > batch.json   # its 2 x NMODES mode cars

start.json: {"name": "chain1", "layout": {"W", "x_front", "d_halo"}, "edits": [...carve.carve edits]}
Every car is the SIZED car (no --skin-mm), half car, fine, 4-core hierarchical, 1000 iterations, legality
gated, carved through carve.carve from its recipe (--carve-json). Mode k is the start car's edits plus one
bump of +-A at centre k on the START CAR's own surface (car_modes.centres, farthest-point order, NMODES 96
so every point is within one radius). A job id is the hash of its recipe: the same car is never timed twice.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "part5-simulation"))

REPO = "anasssssssss1/stem-racing-compute"
CFD = {"decompose": "hierarchical", "layer_overrides": {"nSurfaceLayers": 0}}


def args(carve: dict, cfd: dict) -> list:
    return ["--carve-json", json.dumps(carve, sort_keys=True), "--support", "beam", "--cfd", "--cfd-if-legal",
            "--res", "fine", "--np", "4", "--cfd-iters", "1000", "--keep-runs",
            "--cfd-json", json.dumps(cfd, sort_keys=True)]


def job(stage: str, ref: str, carve: dict, cfd: dict, fetch: list, save_case: bool, prio: int) -> dict:
    a = args(carve, cfd)
    h = hashlib.sha256(json.dumps([ref, carve], sort_keys=True).encode()).hexdigest()[:20]   # = farm.rid: one car, one id
    return {"id": f"p5-{h}", "stage": stage, "prio": prio,
            "recipe": {"repo": REPO, "ref": ref, "args": a, "fetch": fetch, "save_case": save_case}}


def centres_mm(start: dict) -> list:
    import paths  # noqa: F401  (puts the other parts on the path)
    import carve
    import car_modes as cm
    g, rep = carve.carve(start["layout"], start["edits"])
    if any(r.get("status") != "ok" for r in rep):
        raise SystemExit(f"start car refused: {rep}")
    _, V, N = cm.surface(g)
    return [[round(float(v), 4) for v in c] for c in cm.centres(V, N) * 1e3]


def main(argv):
    kind, start, ref = argv[0], json.loads(Path(argv[1]).read_text()), argv[2]
    assert len(ref) == 40, "ref must be a full commit SHA"
    stage = f"p3-map-{start['name']}"
    base = {"layout": start["layout"], "edits": start["edits"]}
    if kind == "base":
        jobs = [job(stage, ref, base, CFD, [], True, 8)]
    else:
        url, sha = argv[3], argv[4]
        cfd = {**CFD, "morph_from": "/tmp/base", "warm_from": "/tmp/base"}
        a_mm, r_mm = float(os.environ.get("P3_MODE_A_MM", "1")), float(os.environ.get("P3_MODE_R_MM", "10"))
        jobs = []
        for k, c in enumerate(centres_mm(start)):
            for s in (1.0, -1.0):
                edits = start["edits"] + [{"bumps": [{"c_mm": c, "amp_mm": s * a_mm, "sigma_mm": r_mm}], "unit": False}]
                j = job(stage, ref, {"layout": start["layout"], "edits": edits}, cfd,
                        [{"url": url, "sha256": sha, "to": "/tmp/base"}], False, 7)
                j["mode"] = f"m{k}{'+' if s > 0 else '-'}"
                jobs.append(j)
    print(json.dumps({"batch": {"id": f"{stage}-{kind}", "label": f"Part 3 compass map {start['name']}: {kind}",
                                "stage": stage, "cars": len(jobs)}, "jobs": jobs}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
