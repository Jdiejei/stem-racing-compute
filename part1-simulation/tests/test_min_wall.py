"""The 3.0 mm machined wall (team limit, 2026-11-14): thin_wall_mask finds a 2.9 mm wall
wherever it sits on the grid, passes a 3.0 mm one, does not call a convex edge a wall, and a
carve never leaves a new wall under 3.0 mm."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

D = 1e-3                                   # 1 mm cells, metres


def _slab(t_mm, z0_mm):
    z = (np.arange(24) + 0.5) * D
    c = (z0_mm + t_mm / 2.0) * 1e-3
    phi = np.abs(z - c) - t_mm * 1e-3 / 2.0
    return np.broadcast_to(phi, (8, 8, 24)).copy()


@pytest.mark.parametrize("z0", [5.0, 5.25, 5.5, 5.75])
def test_wall_29_fails_30_passes(z0):
    from phi_updater import thin_wall_mask
    assert thin_wall_mask(_slab(2.9, z0), D, 3.0e-3).any()
    assert not thin_wall_mask(_slab(3.0, z0), D, 3.0e-3).any()
    assert not thin_wall_mask(_slab(6.0, z0), D, 3.0e-3).any()


def test_oblique_wall():
    """A 2.9 mm wall at 30 deg to the grid is still found; a 3.2 mm one is not."""
    from phi_updater import thin_wall_mask
    X, Y, Z = np.meshgrid(*[(np.arange(30) + 0.5) * D] * 3, indexing="ij")
    nrm = np.array([0.0, np.sin(np.radians(30)), np.cos(np.radians(30))])
    s = X * nrm[0] + Y * nrm[1] + Z * nrm[2] - 15e-3
    for t, want in ((2.9, True), (3.2, False)):
        assert thin_wall_mask(np.abs(s) - t * 1e-3 / 2, D, 3.0e-3).any() == want


def test_convex_edge_is_not_a_wall():
    from phi_updater import thin_wall_mask
    X, Y, Z = np.meshgrid(*[(np.arange(20) + 0.5) * D] * 3, indexing="ij")
    phi = np.maximum(Y - 10e-3, Z - 10e-3)              # solid where y < 10 and z < 10 mm
    out = np.hypot(np.maximum(Y - 10e-3, 0), np.maximum(Z - 10e-3, 0))
    phi = np.where(phi > 0, out, phi)
    assert not thin_wall_mask(phi, D, 3.0e-3).any()


def test_carve_leaves_no_new_thin_wall():
    """Hard thinning through carve.carve (2 mm grid): no wall under 3.0 mm that the start
    block did not already have."""
    import carve
    import cutter
    from geometry_contract import GRID_SPACING_M
    from phi_updater import thin_wall_mask
    lay = {"W": 120.3, "x_front": 46.0, "d_halo": 43.72}
    g0 = cutter.start_block(lay, spacing_mm=2.0)
    t0 = thin_wall_mask(g0.phi.grid, GRID_SPACING_M, 3.0e-3)
    g, reps = carve.carve(lay, [{"bias_mm": -2.5, "trust_mm": 2.0}] * 4, spacing_mm=2.0)
    assert not (thin_wall_mask(g.phi.grid, GRID_SPACING_M, 3.0e-3) & ~t0).any()
    assert all(r["status"] == "ok" for r in reps), reps          # a thinning cut must move, not be refused


def test_finished_body_reports_and_gates_its_wall(tmp_path):
    """cutter.finish measures the sized, milled body's thinnest wall and refuses it under
    3.0 mm; sizing never searched a skin below the wall-legal one."""
    import carve
    import cutter
    lay = {"W": 120.3, "x_front": 46.0, "d_halo": 43.72}
    g, _ = carve.carve(lay, [{"bias_mm": -1.5, "trust_mm": 2.0}], spacing_mm=2.0)
    res = cutter.finish(g, lay, target_body_g=50.0, out_dir=tmp_path)
    w = res["min_machined_wall_mm"]
    assert w is not None
    assert (res["status"] != "ok") or w >= 3.0, res
    if res["status"] == "refused" and res["reason"].startswith("min_wall"):
        assert w < 3.0
    tried = res["report"]["sizing_tried"]
    assert min(t[0] for t in tried) >= res["report"]["skin_lo_mm"] - 1e-9


@pytest.mark.parametrize("mode", ["compare", "freeze"])
def test_guard_a_ridge_that_moved_is_not_a_new_wall(mode):
    """A 2.9 mm wall that was already there and only slid 0.6 mm (its ridge moves a cell)
    is not a new thin wall: the old index check refused every cut on the start block."""
    from phi_updater import wall_guard
    a, b = _slab(2.9, 5.0), _slab(2.9, 5.6)
    g, ok = wall_guard(a, b, D, 3.0e-3, mode=mode)
    assert ok and np.array_equal(g, b)


@pytest.mark.parametrize("mode", ["compare", "freeze"])
def test_guard_stops_a_wall_going_under_3(mode):
    from phi_updater import thin_wall_mask, wall_guard
    a, b = _slab(3.4, 8.0), _slab(2.6, 8.4)              # both faces move in: 3.4 -> 2.6 mm
    g, ok = wall_guard(a, b, D, 3.0e-3, mode=mode)
    assert not thin_wall_mask(g, D, 3.0e-3).any()
    assert ok == (mode == "freeze") and (ok or np.array_equal(g, a))


def test_freeze_keeps_the_move_away_from_the_wall():
    """Freeze undoes the sub-step only around the wall: a face 15 mm away keeps its move."""
    from phi_updater import thin_wall_mask, wall_guard
    z = (np.arange(40) + 0.5) * D
    thin = np.abs(z - 6.7e-3) - 1.7e-3                                  # 3.4 mm wall at 5.0-8.4 mm
    a = np.broadcast_to(np.minimum(thin, 20e-3 - z), (8, 8, 40)).copy()   # + a block from z 20 mm up
    b = a.copy()
    b[..., :12] = np.broadcast_to(np.abs(z[:12] - 6.7e-3) - 1.3e-3, (8, 8, 12))   # wall -> 2.6 mm
    b[..., 14:] -= 0.8e-3                                               # its face grows 0.8 mm
    g, ok = wall_guard(a, b, D, 3.0e-3, mode="freeze")
    assert ok and not thin_wall_mask(g, D, 3.0e-3).any()
    assert np.array_equal(g[..., 14:], b[..., 14:]), "the far move was undone too"


@pytest.mark.parametrize("amp", [1.0, -1.0])
def test_a_1mm_bump_on_the_start_block_moves_it_and_keeps_every_wall(amp):
    """A +-1 mm bump (sigma 10 mm, speed in mm, trust 1 mm) on the top of the start block
    moves the surface there by about 1 mm the right way, leaves it alone 40 mm away, and
    makes no wall thinner than 3.0 mm that was not already that thin."""
    import carve
    import cutter
    from geometry_contract import GRID_SPACING_M
    from phi_updater import _thinned
    from unified_phi import extract_half_surface
    lay = {"W": 120.3, "x_front": 46.0, "d_halo": 43.72}
    g0 = cutter.start_block(lay, spacing_mm=2.0)
    V = np.asarray(extract_half_surface(g0).vertices) * 1e3
    mid = 0.5 * (V[:, 0].min() + V[:, 0].max())
    top = V[(np.abs(V[:, 0] - mid) < 10) & (V[:, 1] < 5)]
    c = top[top[:, 2].argmax()]
    g, reps = carve.carve(lay, [{"bumps": [{"c_mm": c.tolist(), "amp_mm": amp, "sigma_mm": 10.0}],
                                 "unit": False, "trust_mm": 1.0}], spacing_mm=2.0)
    assert reps[0]["status"] == "ok", reps
    d = (g.phi.grid - g0.phi.grid) * 1e3                 # mm; growing lowers phi
    ijk = lambda p: tuple(np.clip(np.rint((p * 1e-3 - np.asarray(g0.region.origin_m)) / GRID_SPACING_M).astype(int),  # noqa: E731
                                  0, np.array(d.shape) - 1))
    moved = -d[ijk(c)]
    assert 0.5 <= moved * np.sign(amp) <= 1.5, f"moved {moved:+.2f} mm at the bump for amp {amp:+.1f}"
    far = c + np.array([-40.0, 0.0, 0.0])
    assert abs(d[ijk(far)]) < 0.25, f"{d[ijk(far)]:+.2f} mm 40 mm away"
    assert not _thinned(g0.phi.grid, g.phi.grid, GRID_SPACING_M, 3.0e-3).any()
