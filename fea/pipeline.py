"""High-level API: build -> modal/static -> post-process. Used by the MCP server and scripts."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from .geometry import MountParams, MATERIALS, build_mesh, mesh_mass_volume
from .inp import write_inp
from .solver import run_calculix, parse_dat
from .export import export_static_vtu, export_modal_vtu


class MountModel:
    def __init__(self, params: MountParams | None = None, workdir="runs/default"):
        self.p = params or MountParams()
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mesh = None
        self.modal = None
        self.static = None

    # ---------------- mesh
    def build(self):
        self.mesh = build_mesh(self.p)
        m = self.mesh
        vol = mesh_mass_volume(m["nodes"], m["tets"])
        mat = MATERIALS[self.p.material]
        return dict(n_nodes=len(m["nodes"]), n_elements=len(m["tets"]),
                    volume_mm3=round(vol, 1), structure_mass_g=round(vol * mat["rho"] * 1e6, 2),
                    n_motor_hole_nodes=len(m["sets"]["MOTORHOLES"]),
                    n_base_hole_nodes=len(m["sets"]["BASEHOLES"]))

    def _need_mesh(self):
        if self.mesh is None:
            self.build()

    # ---------------- modal
    def run_modal(self, n_modes=6, motor_mass_g=60.0):
        self._need_mesh()
        m, mat = self.mesh, MATERIALS[self.p.material]
        mm = m["sets"]["MOTORHOLES"]
        pm = [(n, motor_mass_g * 1e-6 / len(mm)) for n in mm]  # g -> tonne, spread over bolt-hole nodes
        inp = self.dir / "modal.inp"
        write_inp(inp, m["nodes"], m["tets"], m["sets"], mat, f"modal:{n_modes}",
                  boundary=[("BASEHOLES", 1, 3)], point_masses=pm)
        res = parse_dat(run_calculix(inp))
        freqs = [f for _, f in sorted(res["eigen"])][:n_modes]
        disp = [arr[:, 1:4] for (kind, nset, t), arr in res["blocks"].items()
                if kind.startswith("displacements") and nset == "NALL"]  # dict keeps file order = mode order
        shapes = {i + 1: a for i, a in enumerate(disp)}
        self.modal = dict(freqs=freqs, shapes=shapes)
        return dict(frequencies_hz=[round(f, 1) for f in freqs], motor_mass_g=motor_mass_g,
                    note="Motor mass lumped on bolt-hole nodes; base-bolt holes fully fixed.")

    # ---------------- static
    def run_static(self, thrust_N=20.0, torque_Nm=0.4, load_factor=1.0):
        """Thrust pulls wall toward -x; motor reaction torque about x applied as tangential bolt-hole forces."""
        self._need_mesh()
        m, mat = self.mesh, MATERIALS[self.p.material]
        T, M = thrust_N * load_factor, torque_Nm * load_factor * 1000.0  # N, N*mm
        mm = m["sets"]["MOTORHOLES"]
        n = len(mm)
        tang = []
        for nid in mm:
            x, y, z = m["nodes"][nid - 1]
            ry, rz = y, z - self.p.motor_z
            r = math.hypot(ry, rz)
            f = M / (n * r) if r > 1e-6 else 0.0
            tang.append((-f * rz / r, f * ry / r) if r > 1e-6 else (0.0, 0.0))
        tang = np.array(tang)
        tang -= tang.mean(axis=0)  # pure couple: remove spurious net force from uneven node spacing
        cl = []
        for nid, (fy, fz) in zip(mm, tang):
            cl += [(nid, 1, -T / n), (nid, 2, fy), (nid, 3, fz)]
        inp = self.dir / "static.inp"
        write_inp(inp, m["nodes"], m["tets"], m["sets"], mat, "static",
                  boundary=[("BASEHOLES", 1, 3)], cloads=cl)
        res = parse_dat(run_calculix(inp))
        U = S = RF = None
        for (kind, nset, t), arr in res["blocks"].items():
            if kind.startswith("displacements") and nset == "NALL":
                U = arr[:, 1:4]
            elif kind.startswith("stresses"):
                S = arr
            elif kind.startswith("total force"):
                RF = arr.ravel()[:3]
        # von Mises per element (mean of integration points)
        s = S[:, 2:8]
        sxx, syy, szz, sxy, sxz, syz = s.T
        vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                     + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2))
        el = S[:, 0].astype(int)
        nel = len(m["tets"])
        vm_el = np.zeros(nel)
        cnt = np.zeros(nel)
        np.add.at(vm_el, el - 1, vm)
        np.add.at(cnt, el - 1, 1)
        vm_el /= np.maximum(cnt, 1)
        # nodal VM (average of adjacent elements) for plotting
        vm_n = np.zeros(len(m["nodes"]))
        c = np.zeros(len(m["nodes"]))
        for k in range(4):
            np.add.at(vm_n, m["tets"][:, k] - 1, vm_el)
            np.add.at(c, m["tets"][:, k] - 1, 1)
        vm_n /= np.maximum(c, 1)
        umag = np.linalg.norm(U, axis=1)
        self.static = dict(U=U, vm_el=vm_el, vm_n=vm_n)
        applied = np.array([-T, 0, 0])
        return dict(max_displacement_mm=round(float(umag.max()), 4),
                    max_vonmises_MPa=round(float(vm_el.max()), 1),
                    yield_MPa=mat["yield_"],
                    safety_factor=round(mat["yield_"] / float(vm_el.max()), 2),
                    reaction_force_N=[round(float(v), 3) for v in RF],
                    applied_force_N=applied.tolist(),
                    equilibrium_error_N=round(float(np.linalg.norm(RF + applied)), 4),
                    note="Peak stress at re-entrant corner / bolt holes is mesh-sensitive (singular); "
                         "judge with mesh_convergence.")

    # ---------------- studies
    def mesh_convergence(self, sizes=(4.0, 3.0, 2.0, 1.5), **kw):
        out = []
        keep = (self.p.mesh_size, self.p.hole_mesh_size)
        for h in sizes:
            self.p.mesh_size, self.p.hole_mesh_size = h, h * 0.4
            info = self.build()
            mo = self.run_modal(n_modes=3, **{k: v for k, v in kw.items() if k == "motor_mass_g"})
            st = self.run_static()
            out.append(dict(mesh_size=h, n_elements=info["n_elements"], f1_hz=mo["frequencies_hz"][0],
                            max_disp_mm=st["max_displacement_mm"], max_vm_MPa=st["max_vonmises_MPa"]))
        self.p.mesh_size, self.p.hole_mesh_size = keep
        self.mesh = None
        return out

    def parameter_sweep(self, name, values, thrust_N=20.0, torque_Nm=0.4, motor_mass_g=60.0):
        out = []
        orig = getattr(self.p, name)
        for v in values:
            setattr(self.p, name, float(v))
            info = self.build()
            mo = self.run_modal(n_modes=3, motor_mass_g=motor_mass_g)
            st = self.run_static(thrust_N, torque_Nm)
            out.append(dict(**{name: v}, mass_g=info["structure_mass_g"], f1_hz=mo["frequencies_hz"][0],
                            max_vm_MPa=st["max_vonmises_MPa"], safety_factor=st["safety_factor"],
                            max_disp_mm=st["max_displacement_mm"]))
        setattr(self.p, name, orig)
        self.mesh = None
        return out
