"""Export results to VTU (ParaView). Open with ParaView: File > Open > *.vtu, then
colour by 'von_Mises_MPa' / 'U_mag_mm' and use Filters > Warp By Vector (U) for deformation."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def _mesh(model, point_data):
    import meshio
    m = model.mesh
    # Our node order for C3D10 equals VTK quadratic tetra order.
    return meshio.Mesh(m["nodes"], [("tetra10", m["tets"] - 1)], point_data=point_data)


def export_static_vtu(model, path):
    """Static results: displacement vector, |U|, nodal von Mises."""
    s = model.static
    if s is None:
        raise ValueError("run_static first")
    U = np.asarray(s["U"], dtype=float)
    data = {"U": U, "U_mag_mm": np.linalg.norm(U, axis=1), "von_Mises_MPa": np.asarray(s["vm_n"], dtype=float)}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    _mesh(model, data).write(path)
    return str(path)


def export_modal_vtu(model, path):
    """Modal results: one displacement vector field per mode (mode_1 ... mode_N). Frequencies in field names."""
    md = model.modal
    if md is None:
        raise ValueError("run_modal first")
    data = {}
    for k, U in md["shapes"].items():
        f = md["freqs"][k - 1]
        data[f"mode_{k}_{f:.0f}Hz"] = np.asarray(U, dtype=float)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    _mesh(model, data).write(path)
    return str(path)
