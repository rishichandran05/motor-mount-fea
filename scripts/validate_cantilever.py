"""Validate the Gmsh -> .inp -> CalculiX -> parse chain against Euler-Bernoulli cantilever theory."""
import math, sys, tempfile
from pathlib import Path
import numpy as np, gmsh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fea.inp import write_inp
from fea.solver import run_calculix, parse_dat
from fea.geometry import MATERIALS

L, b, h = 100.0, 10.0, 5.0
mat = MATERIALS["AL6061-T6"]
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.occ.addBox(0, 0, 0, L, b, h); gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.5); gmsh.option.setNumber("Mesh.ElementOrder", 2)
gmsh.model.mesh.generate(3)
nt, nc, _ = gmsh.model.mesh.getNodes(); nc = nc.reshape(-1, 3)
et, _, en = gmsh.model.mesh.getElements(3)
tets = np.array(en[list(et).index(11)], dtype=int).reshape(-1, 10); tets[:, [8, 9]] = tets[:, [9, 8]]
used = np.unique(tets); rm = {int(t): i + 1 for i, t in enumerate(used)}; ix = {int(t): i for i, t in enumerate(nt)}
nodes = np.array([nc[ix[int(t)]] for t in used]); tets = np.vectorize(rm.get)(tets)
fixed = [i + 1 for i, p in enumerate(nodes) if p[0] < 1e-6]
gmsh.finalize()
d = Path(tempfile.mkdtemp()); inp = d / "cant.inp"
write_inp(inp, nodes, tets, {"BASEHOLES": fixed}, mat, "modal:4", boundary=[("BASEHOLES", 1, 3)])
f = [fr for _, fr in sorted(parse_dat(run_calculix(inp))["eigen"])]
I = b * h ** 3 / 12; A = b * h; E = mat["E"]; rho = mat["rho"]
f1 = (1.8751 ** 2 / (2 * math.pi)) * math.sqrt(E * I / (rho * A * L ** 4))
print(f"FEA modes (Hz): {[round(x,1) for x in f]}")
print(f"Euler-Bernoulli bending mode 1 (h-direction): {f1:.1f} Hz")
print(f"closest FEA mode error: {min(abs(x-f1)/f1 for x in f)*100:.2f} %")
