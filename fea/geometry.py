"""Parametric fixed-wing / VTOL motor mount: L-bracket + gusset, built and meshed with Gmsh.

Units: mm, tonne, N, s, MPa  (density in t/mm^3).
Coordinates: base plate lies in XY (z=0..base_t), firewall wall in YZ at x=0..wall_t.
Motor bolts on the wall, thrust along -x (tractor). Base bolts clamp to airframe.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import numpy as np
import gmsh


@dataclass
class MountParams:
    base_l: float = 60.0      # base plate length (x)
    base_w: float = 44.0      # base plate / wall width (y)
    base_t: float = 4.0
    wall_h: float = 40.0      # wall height above z=0
    wall_t: float = 4.0
    gusset_t: float = 3.0     # 0 disables gusset
    gusset_l: float = 30.0
    gusset_h: float = 28.0
    motor_pcd: float = 19.0   # motor bolt circle diameter
    motor_hole_d: float = 3.2
    motor_center_d: float = 10.0
    base_hole_d: float = 4.2
    mesh_size: float = 2.0    # global max element size
    hole_mesh_size: float = 0.8
    material: str = "AL6061-T6"

    def to_dict(self):
        return asdict(self)

    @property
    def motor_z(self):
        return self.wall_h - 16.0


MATERIALS = {
    # E [MPa], nu, rho [t/mm^3], yield [MPa]
    "AL6061-T6": dict(E=68900.0, nu=0.33, rho=2.70e-9, yield_=276.0),
    "AL7075-T6": dict(E=71700.0, nu=0.33, rho=2.81e-9, yield_=503.0),
    "PETG": dict(E=2100.0, nu=0.38, rho=1.27e-9, yield_=45.0),
    "PA12-CF": dict(E=6000.0, nu=0.36, rho=1.15e-9, yield_=70.0),
}


def build_mesh(p: MountParams):
    """Return dict(nodes, tets(10-node, Abaqus order), tris, sets{name: node ids}, motor_nodes xyz)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        occ = gmsh.model.occ
        gmsh.model.add("mount")
        base = occ.addBox(0, -p.base_w / 2, 0, p.base_l, p.base_w, p.base_t)
        wall = occ.addBox(0, -p.base_w / 2, 0, p.wall_t, p.base_w, p.wall_h)
        parts = [(3, wall)]
        if p.gusset_t > 0:
            x0, z0 = p.wall_t, p.base_t
            pts = [occ.addPoint(x0, -p.gusset_t / 2, z0),
                   occ.addPoint(x0 + p.gusset_l, -p.gusset_t / 2, z0),
                   occ.addPoint(x0, -p.gusset_t / 2, z0 + p.gusset_h)]
            ls = [occ.addLine(pts[i], pts[(i + 1) % 3]) for i in range(3)]
            s = occ.addPlaneSurface([occ.addCurveLoop(ls)])
            ext = occ.extrude([(2, s)], 0, p.gusset_t, 0)
            parts.append([e for e in ext if e[0] == 3][0])
        body, _ = occ.fuse([(3, base)], parts)

        cuts = []
        r = p.motor_pcd / 2
        for k in range(4):
            a = math.radians(45 + 90 * k)
            cuts.append(occ.addCylinder(-1, r * math.cos(a), p.motor_z + r * math.sin(a),
                                        p.wall_t + 2, 0, 0, p.motor_hole_d / 2))
        cuts.append(occ.addCylinder(-1, 0, p.motor_z, p.wall_t + 2, 0, 0, p.motor_center_d / 2))
        for x in (p.base_l - 8, 22.0):
            for y in (-15.0, 15.0):
                cuts.append(occ.addCylinder(x, y, -1, 0, 0, p.base_t + 2, p.base_hole_d / 2))
        occ.cut(body, [(3, c) for c in cuts])
        occ.synchronize()

        # classify hole surfaces
        sets_surf = {"MOTORHOLES": [], "BASEHOLES": []}
        for dim, tag in gmsh.model.getEntities(2):
            if gmsh.model.getType(dim, tag) != "Cylinder":
                continue
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(dim, tag)
            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
            if cx < p.wall_t + 0.1 and cz > p.base_t and abs(dy - p.motor_hole_d) < 0.4:
                sets_surf["MOTORHOLES"].append(tag)
            elif cz < p.base_t + 0.1 and cx > p.wall_t and abs(dx - p.base_hole_d) < 0.4:
                sets_surf["BASEHOLES"].append(tag)
        assert len(sets_surf["MOTORHOLES"]) >= 4 and len(sets_surf["BASEHOLES"]) >= 4, sets_surf

        # mesh sizing: fine near holes
        hole_curves = []
        for t in sets_surf["MOTORHOLES"] + sets_surf["BASEHOLES"]:
            hole_curves += [c for _, c in gmsh.model.getBoundary([(2, t)], oriented=False)]
        f1 = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f1, "CurvesList", hole_curves)
        gmsh.model.mesh.field.setNumber(f1, "Sampling", 20)
        f2 = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f2, "InField", f1)
        gmsh.model.mesh.field.setNumber(f2, "SizeMin", p.hole_mesh_size)
        gmsh.model.mesh.field.setNumber(f2, "SizeMax", p.mesh_size)
        gmsh.model.mesh.field.setNumber(f2, "DistMin", 1.0)
        gmsh.model.mesh.field.setNumber(f2, "DistMax", 6.0)
        gmsh.model.mesh.field.setAsBackgroundMesh(f2)
        for o in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary"):
            gmsh.option.setNumber("Mesh." + o, 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
        gmsh.model.mesh.generate(3)

        # extract
        ntags, ncoord, _ = gmsh.model.mesh.getNodes()
        ncoord = ncoord.reshape(-1, 3)
        etypes, etags, enodes = gmsh.model.mesh.getElements(3)
        k = list(etypes).index(11)  # 10-node tet
        tets = np.array(enodes[k], dtype=np.int64).reshape(-1, 10)
        # gmsh tet10 -> Abaqus C3D10: swap last two mid-nodes
        tets[:, [8, 9]] = tets[:, [9, 8]]
        used = np.unique(tets)
        remap = {int(t): i + 1 for i, t in enumerate(used)}
        idx = {int(t): i for i, t in enumerate(ntags)}
        nodes = np.array([ncoord[idx[int(t)]] for t in used])
        tets = np.vectorize(remap.get)(tets)

        sets = {}
        for name, tags in sets_surf.items():
            ids = set()
            for t in tags:
                nt, _, _ = gmsh.model.mesh.getNodes(2, t, includeBoundary=True)
                ids.update(remap[int(n)] for n in nt if int(n) in remap)
            sets[name] = sorted(ids)

        # boundary triangles (corner nodes) for plotting
        tris = []
        for _, t in gmsh.model.getEntities(2):
            et, _, en = gmsh.model.mesh.getElements(2, t)
            for a, b in zip(et, en):
                if a == 9:  # 6-node triangle
                    arr = np.array(b, dtype=np.int64).reshape(-1, 6)[:, :3]
                    tris.append(np.vectorize(remap.get)(arr))
        tris = np.vstack(tris)
    finally:
        gmsh.finalize()
    return dict(nodes=nodes, tets=tets, tris=tris, sets=sets)


def mesh_mass_volume(nodes, tets):
    """Volume (mm^3) from corner nodes of tets."""
    c = nodes[tets[:, :4] - 1]
    v = np.einsum("ij,ij->i", c[:, 1] - c[:, 0], np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0])) / 6.0
    return float(np.abs(v).sum())
