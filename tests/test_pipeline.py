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
