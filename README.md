# Agentic FEA: Parametric Motor Mount via Claude + MCP + CalculiX

Claude drives a full open-source FEA loop through the Model Context Protocol:
**Gmsh (CAD + 10-node tet mesh) → Abaqus-format `.inp` → CalculiX → parsed results → plots**.
No commercial licence needed. Ask in plain language, e.g. *"Make the wall thinner until f1 drops below 1100 Hz, then tell me the lightest design that keeps safety factor > 5 at 60 N thrust."*

**Part:** fixed-wing / VTOL motor mount (L-bracket + gusset, 4× motor bolts on a 19 mm circle, 4× base bolts), Al 6061-T6.
**Analyses:** modal (motor mass lumped on bolt nodes) and static (thrust + reaction torque), parameter sweep, mesh convergence.

## Quick start
```bash
sudo apt install calculix-ccx          # or set CCX=/path/to/ccx
pip install -r requirements.txt
python scripts/validate_cantilever.py  # solver-chain check vs beam theory
python scripts/run_demo.py             # build, modal, static, plots in assets/
pytest
```
Claude Code: the repo ships `.mcp.json`. Claude Desktop: add
`{"command": "python", "args": ["/abs/path/mcp_server/server.py"]}` under `mcpServers`.

## MCP tools
`set_design` · `get_design` · `run_modal` · `run_static` · `plot_mode_shape` · `plot_stress_field` · `parameter_sweep` · `mesh_convergence`

## Verification (measured in this repo)
| Check | Result |
|---|---|
| Cantilever 100×10×5 mm, f1 vs Euler–Bernoulli | 409.6 vs 408.0 Hz (0.4 %) |
| Reaction force vs applied thrust (20 N) | 20.000 N, error 0.000 N |
| Linearity (2× load → 2× displacement) | pass (`pytest`) |
| f1 mesh convergence, 4824 → 51269 tets | 1181.5 → 1173.1 Hz (0.7 % spread) |

**Known limitation:** peak von Mises at the fixed base-bolt holes keeps rising with refinement (6.4 → 9.7 MPa). That is a stress singularity from fully-fixed hole walls, not a converged value. Use displacement, frequency and stress *trends*; for real stress, model washer-face contact or bolt preload.

## Modelling assumptions
Units mm-t-N-s-MPa · base-hole walls fully fixed · motor = lumped mass spread over motor-bolt nodes (no rigid coupling) · linear elastic, small displacement · no fillets.

## Abaqus
The `.inp` uses Abaqus keywords (`*NODE`, `*ELEMENT C3D10`, `*FREQUENCY`, `*STATIC`, `*CLOAD`, `*NODE PRINT`), so `fea/inp.py` is the porting path. An Abaqus runner and `.dat` parser are **not implemented or tested**; `fea/solver.py` is the one place to add them.

## Layout
```
fea/geometry.py  parametric CAD + mesh      fea/inp.py     .inp writer
fea/solver.py    ccx runner + .dat parser   fea/pipeline.py  MountModel API
fea/plots.py     3D mode/stress plots       mcp_server/server.py  MCP tools
scripts/         demo + validation          tests/         pytest
```
