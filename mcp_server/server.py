"""MCP server exposing the motor-mount FEA pipeline to Claude (Claude Desktop / Claude Code)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp.server.fastmcp import FastMCP, Image
from fea import MountModel, MountParams
from fea.geometry import MATERIALS
from fea.plots import plot_mode, plot_stress

mcp = FastMCP("motor-mount-fea")
RUNS = Path(__file__).resolve().parents[1] / "runs" / "mcp"
STATE = {"model": MountModel(workdir=RUNS)}


def _m() -> MountModel:
    return STATE["model"]


@mcp.tool()
def set_design(**params: float) -> dict:
    """Set geometry/mesh parameters (mm). Keys: base_l, base_w, base_t, wall_h, wall_t, gusset_t (0=off),
    gusset_l, gusset_h, motor_pcd, motor_hole_d, motor_center_d, base_hole_d, mesh_size, hole_mesh_size,
    material (AL6061-T6|AL7075-T6|PETG|PA12-CF). Rebuilds mesh and returns stats."""
    p = _m().p
    for k, v in params.items():
        if not hasattr(p, k):
            return {"error": f"unknown parameter {k}", "valid": list(p.to_dict())}
        setattr(p, k, v if k == "material" else float(v))
    if p.material not in MATERIALS:
        return {"error": f"unknown material {p.material}", "valid": list(MATERIALS)}
    return {"params": p.to_dict(), "mesh": _m().build()}


@mcp.tool()
def get_design() -> dict:
    """Current design parameters."""
    return _m().p.to_dict()


@mcp.tool()
def run_modal(n_modes: int = 6, motor_mass_g: float = 60.0) -> dict:
    """Modal analysis (base-bolt holes fixed, motor mass lumped on motor bolt holes). Returns Hz."""
    return _m().run_modal(n_modes, motor_mass_g)


@mcp.tool()
def run_static(thrust_N: float = 20.0, torque_Nm: float = 0.4, load_factor: float = 1.0) -> dict:
    """Static stress under motor thrust (-x) and reaction torque about x. load_factor scales both."""
    return _m().run_static(thrust_N, torque_Nm, load_factor)


@mcp.tool()
def plot_mode_shape(mode: int = 1) -> Image:
    """Render mode shape of the last modal run as an image."""
    if _m().modal is None:
        raise ValueError("run_modal first")
    return Image(path=plot_mode(_m(), mode, RUNS / f"mode{mode}.png"))


@mcp.tool()
def plot_stress_field() -> Image:
    """Render von Mises stress of the last static run as an image."""
    if _m().static is None:
        raise ValueError("run_static first")
    return Image(path=plot_stress(_m(), RUNS / "stress.png"))


@mcp.tool()
def parameter_sweep(name: str, values: list[float], thrust_N: float = 20.0,
                    torque_Nm: float = 0.4, motor_mass_g: float = 60.0) -> list[dict]:
    """Sweep one parameter (e.g. wall_t, gusset_t). Returns mass, f1, max stress, safety factor per value."""
    return _m().parameter_sweep(name, values, thrust_N, torque_Nm, motor_mass_g)


@mcp.tool()
def mesh_convergence(sizes: list[float] = [4.0, 3.0, 2.0, 1.5]) -> list[dict]:
    """Rerun modal+static at several global mesh sizes to check convergence of f1, displacement, stress."""
    return _m().mesh_convergence(tuple(sizes))


if __name__ == "__main__":
    mcp.run()
