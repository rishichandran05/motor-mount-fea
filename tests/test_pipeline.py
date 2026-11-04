import numpy as np
from fea import MountModel, MountParams


def test_modal_and_static(tmp_path):
    m = MountModel(MountParams(mesh_size=4.0, hole_mesh_size=1.6), workdir=tmp_path)
    info = m.build()
    assert info["n_elements"] > 1000
    f = m.run_modal(3)["frequencies_hz"]
    assert len(f) == 3 and f[0] < f[1] < f[2]
    s = m.run_static(20.0, 0.4)
    big = m.run_static(1000.0, 20.0)
    assert s["equilibrium_error_N"] < 0.05          # reaction == applied thrust
    s2 = m.run_static(2000.0, 40.0)
    assert abs(s2["max_displacement_mm"] / big["max_displacement_mm"] - 2) < 0.01   # linearity


def test_vtu_export_matches_solver(tmp_path):
    import numpy as np, meshio
    from fea.export import export_static_vtu, export_modal_vtu
    m = MountModel(MountParams(mesh_size=4.0, hole_mesh_size=1.6), workdir=tmp_path)
    m.build(); m.run_modal(2); s = m.run_static()
    # mid-edge nodes sit on the VTK quadratic-tet edges (0-1,1-2,2-0,0-3,1-3,2-3)
    X, T = m.mesh["nodes"], m.mesh["tets"] - 1
    for k, (a, b) in enumerate([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]):
        mid = (X[T[:, a]] + X[T[:, b]]) / 2
        assert np.median(np.linalg.norm(X[T[:, 4 + k]] - mid, axis=1)) < 1e-6   # straight edges exact; curved hole edges deviate
    r = meshio.read(export_static_vtu(m, tmp_path / "static.vtu"))
    assert abs(r.point_data["U_mag_mm"].max() - s["max_displacement_mm"]) < 1e-3
    r2 = meshio.read(export_modal_vtu(m, tmp_path / "modal.vtu"))
    assert len(r2.point_data) == 2
