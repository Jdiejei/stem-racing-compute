"""The carving loop on the public job queue: one tick reads results, moves every chain, and writes the
next jobs (run on Oracle by cron or at a check-in, never on the PC; the reviewer adds the batch).

    python farm.py init  state.json start1.json start2.json ... --ref <sha>
    python farm.py tick  state.json results.json out_batch.json --shape-sigma-file <Part 2 fine gate json>

A chain's x is the amplitudes (mm) of NMODES bumps at FIXED centres on its start car's surface, so a car
is start edits + one bump edit carrying sum x_k * bump_k; step(x, d, size) = x + size * d.
The compass at x is a full map of the car x (x +- A on every mode, central differences), taken at the
start and after every KEEP only; a rejected round keeps the same car and the same map (map reuse:
recommended, the user's decision). A car is timed once: case ids with the same recipe share its result
(the solver is deterministic, so copies are byte-identical). Every car is sized, carve-json, fine,
4-core hierarchical, legality gated; each result must be legal with matching hashes (carve_loop._t).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import carve_loop as cl

REPO = "anasssssssss1/stem-racing-compute"
A_MM, R_MM = 1.0, 10.0
CFD = {"decompose": "hierarchical", "layer_overrides": {"nSurfaceLayers": 0}}


def recipe(chain: dict, x) -> dict:
    bumps = [{"c_mm": c, "amp_mm": round(float(a), 6), "sigma_mm": R_MM}
             for c, a in zip(chain["centres"], x) if a != 0]
    edits = chain["start"]["edits"] + ([{"bumps": bumps, "unit": False}] if bumps else [])
    return {"layout": chain["start"]["layout"], "edits": edits}


def rid(ref: str, rec: dict) -> str:
    return "p5-" + hashlib.sha256(json.dumps([ref, rec], sort_keys=True).encode()).hexdigest()[:20]


def job(ref: str, rec: dict, base: dict | None, save: bool) -> dict:
    cfd = dict(CFD)
    fetch = []
    if base:                                     # reshape and warm-start from the car this one is near
        cfd.update(morph_from="/tmp/base", warm_from="/tmp/base")
        fetch = [{"url": base["case_url"], "sha256": base.get("case_sha256", ""), "to": "/tmp/base"}]
    args = ["--carve-json", json.dumps(rec, sort_keys=True), "--support", "beam", "--cfd", "--cfd-if-legal",
            "--res", "fine", "--np", "4", "--cfd-iters", "1000", "--keep-runs", "--cfd-json", json.dumps(cfd, sort_keys=True)]
    return {"id": rid(ref, rec), "stage": "p5-loop", "prio": 6,
            "recipe": {"repo": REPO, "ref": ref, "args": args, "fetch": fetch, "save_case": save}}


def init(state_path: Path, starts: list, ref: str, centres: list) -> dict:
    assert len(ref) == 40
    st = {"ref": ref, "chains": []}
    for i, (s, C) in enumerate(zip(starts, centres)):
        c = cl.new_chain(i, [0.0] * len(C), 0.5)
        c.update(start=s, centres=C, maps={})
        st["chains"].append(c)
    state_path.write_text(json.dumps(st))
    return st


def _val(results: dict, ref: str, rec: dict):
    r = results.get(rid(ref, rec))
    return None if r is None or r.get("status") != "done" else r["result"]


def tick(st: dict, results: dict, shape_sigma: float) -> list:
    """Move every chain as far as the results allow; return the jobs still missing."""
    ref, want = st["ref"], {}
    ready = []
    for c in st["chains"]:
        if c["status"] != "running":
            continue
        key = json.dumps([round(v, 6) for v in c["x"]])
        cur = recipe(c, c["x"])
        base = _val(results, ref, cur)
        if base is None:
            want[rid(ref, cur)] = job(ref, cur, None, True)          # the current car, its case saved
            continue
        if key not in c["maps"]:                                     # map the current car (start, after a keep)
            g, missing = [], False
            for k in range(len(c["x"])):
                T = []
                for s in (1, -1):
                    x = list(c["x"]); x[k] += s * A_MM
                    rec = recipe(c, x)
                    v = _val(results, ref, rec)
                    if v is None:
                        want[rid(ref, rec)] = job(ref, rec, base, False); missing = True
                    else:
                        T.append(cl._t(v))
                if len(T) == 2:
                    g.append((T[0] - T[1]) / (2 * A_MM))
            if missing:
                continue
            c["maps"] = {key: g}                                     # only the current car's map is kept
        ready.append(c)
    view = {"chains": ready}
    cl.plan(view, lambda x: np.asarray(next(c["maps"][json.dumps([round(v, 6) for v in x])]
                                            for c in ready if json.dumps([round(v, 6) for v in x]) in c["maps"])),
            lambda x, d, size: [float(v) for v in np.asarray(x) + size * np.asarray(d)])
    res = {}
    for case_id, (x, _) in cl.needed(view).items():
        c = next(c for c in ready if case_id.split("_")[1] == f"c{c['id']}")
        rec = recipe(c, x)
        v = _val(results, ref, rec)
        if v is None:
            want[rid(ref, rec)] = job(ref, rec, _val(results, ref, recipe(c, c["x"])), True)
        else:
            res[case_id] = v                                         # every copy of one car gets its one result
    cl.advance(view, res, prior_sigma=shape_sigma, shape_sigma=shape_sigma)
    return list(want.values())


def main(argv):
    if argv[0] == "init":                       # centres on each start car's own surface (needs the cutter)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part3-simulation" / "checks"))
        import map_jobs
        starts = [json.loads(Path(p).read_text()) for p in argv[2:argv.index("--ref")]]
        init(Path(argv[1]), starts, argv[argv.index("--ref") + 1], [map_jobs.centres_mm(s) for s in starts])
    elif argv[0] == "tick":
        st = json.loads(Path(argv[1]).read_text())
        results = json.loads(Path(argv[2]).read_text())
        shape = json.loads(Path(argv[argv.index("--shape-sigma-file") + 1]).read_text())["shape_sigma_s"]
        jobs = tick(st, results, float(shape))
        Path(argv[1]).write_text(json.dumps(st))
        Path(argv[3]).write_text(json.dumps({"batch": {"id": f"p5-loop-{len(jobs)}", "stage": "p5-loop",
                                                       "cars": len(jobs)}, "jobs": jobs}, indent=1))
        print(f"LAB farm: {len(jobs)} jobs; chains " + ", ".join(f"{c['id']} {c['status']} r{c['round']}"
                                                                  for c in st["chains"]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
