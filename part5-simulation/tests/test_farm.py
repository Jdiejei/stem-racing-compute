"""The queue driver on a fake queue: every car is timed once (copies share one result), maps are taken
only at the start and after a keep, and chains on a deterministic stopwatch with shape noise reach the
minimum of a bowl without keeping a step that is not faster beyond the bar."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import farm

REF = "0" * 40


def _start(i):
    return {"name": f"c{i}", "layout": {"W": 120.3, "x_front": 46.0, "d_halo": 43.72}, "edits": []}


def _amps(rec):
    b = rec["edits"][-1]["bumps"] if rec["edits"] else []
    return b


def test_farm_on_a_fake_queue(tmp_path):
    centres = [[float(k), 0.0, 0.0] for k in range(3)]
    st = farm.init(tmp_path / "s.json", [_start(0), _start(1)], REF, [centres, centres])
    opt = np.array([2.0, -1.0, 0.5])
    shape = 0.3e-3
    timed = {}

    def T(rec):
        x = np.zeros(3)
        for b in _amps(rec):
            x[int(b["c_mm"][0])] = b["amp_mm"]
        h = int(hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:8], 16)
        return 1.0 + 1e-3 * float(np.sum((x - opt) ** 2)) + shape * np.random.default_rng(h).normal()

    results = {}
    for _ in range(3000):
        jobs = farm.tick(st, results, shape)
        if not jobs and all(c["status"] != "running" for c in st["chains"]):
            break
        for j in jobs:
            rec = json.loads(j["recipe"]["args"][1])
            assert j["id"] not in timed, "a car was asked for twice"
            timed[j["id"]] = 1
            results[j["id"]] = {"status": "done", "result": {"legal": True, "T_s": T(rec), "expect_field_sha": "a",
                                                             "field_sha": "a", "judged_field_sha": "a",
                                                             "case_url": "u"}}
    for c in st["chains"]:
        assert c["status"] != "running"
        assert len(c["maps"]) <= 1
        x = np.asarray(c["x"])
        assert float(np.sum((x - opt) ** 2)) < float(np.sum(opt ** 2))      # better than the start
        assert all(h["gain_confirm"] > h["bar"] for h in c["history"] if h["kept"])
