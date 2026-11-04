"""Write Abaqus-format .inp (also read by CalculiX). Units mm-t-N-s-MPa."""
from __future__ import annotations
import numpy as np


def _chunks(ids, n=8):
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


def write_inp(path, nodes, tets, nsets, mat, steps, boundary=(), cloads=(), point_masses=()):
    """nsets: {name: [ids]}; boundary: [(nset, dof1, dof2)];
    cloads: [(node, dof, value)]; point_masses: [(node, mass_t)];
    steps: 'modal:N' or 'static'."""
    L = ["*HEADING", "motor-mount model (generated)", "*NODE"]
    L += [f"{i + 1}, {x:.6f}, {y:.6f}, {z:.6f}" for i, (x, y, z) in enumerate(nodes)]
    L.append("*ELEMENT, TYPE=C3D10, ELSET=EALL")
    L += [f"{i + 1}, " + ", ".join(map(str, t)) for i, t in enumerate(tets)]
    L.append("*NSET, NSET=NALL, GENERATE")
    L.append(f"1, {len(nodes)}, 1")
    for name, ids in nsets.items():
        L.append(f"*NSET, NSET={name}")
        L += [", ".join(map(str, c)) for c in _chunks(list(ids))]
    if point_masses:
        base = len(tets)
        L.append("*ELEMENT, TYPE=MASS, ELSET=EMASS")
        L += [f"{base + i + 1}, {n}" for i, (n, _) in enumerate(point_masses)]
        # all point masses share one value in this generator
        L.append("*MASS, ELSET=EMASS")
        L.append(f"{point_masses[0][1]:.9e}")
    L += ["*MATERIAL, NAME=MAT", "*ELASTIC", f"{mat['E']}, {mat['nu']}",
          "*DENSITY", f"{mat['rho']:.4e}",
          "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT"]
    L.append("*BOUNDARY")
    for ns, d1, d2 in boundary:
        L.append(f"{ns}, {d1}, {d2}")
    if steps.startswith("modal"):
        n = int(steps.split(":")[1])
        L += ["*STEP", "*FREQUENCY", f"{n}",
              "*NODE PRINT, NSET=NALL", "U", "*END STEP"]
    else:
        L += ["*STEP", "*STATIC"]
        if cloads:
            L.append("*CLOAD")
            L += [f"{n}, {d}, {v:.8e}" for n, d, v in cloads]
        L += ["*NODE PRINT, NSET=NALL", "U",
              "*NODE PRINT, NSET=BASEHOLES, TOTALS=ONLY", "RF",
              "*EL PRINT, ELSET=EALL", "S", "*END STEP"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
