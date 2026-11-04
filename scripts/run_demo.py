import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fea import MountModel
from fea.plots import plot_mode, plot_stress

out = Path(__file__).resolve().parents[1] / "assets"; out.mkdir(exist_ok=True)
m = MountModel(workdir="runs/demo")
print(json.dumps(m.build(), indent=1))
print(json.dumps(m.run_modal(6), indent=1))
print(json.dumps(m.run_static(), indent=1))
plot_mode(m, 1, out / "mode1.png"); plot_mode(m, 2, out / "mode2.png"); plot_stress(m, out / "stress.png")
conv = m.mesh_convergence()
print("convergence", json.dumps(conv, indent=1))
json.dump(conv, open(out / "convergence.json", "w"), indent=1)
sweep = m.parameter_sweep("wall_t", [3, 4, 5, 6])
print("sweep wall_t", json.dumps(sweep, indent=1))
json.dump(sweep, open(out / "sweep_wall_t.json", "w"), indent=1)
