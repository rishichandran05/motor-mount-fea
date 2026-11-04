"""Solver adapters. CalculiX is implemented; Abaqus is a documented porting path (not tested)."""
from __future__ import annotations
import os, re, shutil, subprocess
from pathlib import Path
import numpy as np


class SolverError(RuntimeError):
    pass


def run_calculix(inp: Path, timeout=600, ccx=None, threads=None):
    ccx = ccx or os.environ.get("CCX") or shutil.which("ccx") or shutil.which("ccx_2.21")
    if not ccx:
        raise SolverError("CalculiX 'ccx' not found. Install: apt install calculix-ccx (or set CCX).")
    inp = Path(inp)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads or os.cpu_count() or 1))
    r = subprocess.run([ccx, "-i", inp.stem], cwd=inp.parent, capture_output=True,
                       text=True, timeout=timeout, env=env)
    dat = inp.with_suffix(".dat")
    if r.returncode != 0 or not dat.exists():
        raise SolverError(f"ccx failed (rc={r.returncode}):\n{r.stdout[-1500:]}\n{r.stderr[-500:]}")
    return dat


_HDR = re.compile(r"^\s*([a-z ]+?)\s*\(([^)]*)\)\s*for set\s+(\w+)\s+and time\s+([0-9.eE+-]+)", re.I)


def parse_dat(dat: Path):
    """Return dict: blocks[(kind,set,time)] -> ndarray; eigen -> [(mode, freq_hz)]."""
    blocks, eigen = {}, []
    lines = Path(dat).read_text().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if "E I G E N V A L U E" in ln:
            i += 1
            while i < len(lines):
                t = lines[i].split()
                if len(t) == 5 and t[0].isdigit():
                    try:
                        eigen.append((int(t[0]), float(t[3])))
                    except ValueError:
                        break
                elif eigen and not lines[i].strip():
                    break
                i += 1
            continue
        m = _HDR.match(ln)
        if m:
            kind = m.group(1).strip().lower()
            key = (kind, m.group(3).upper(), float(m.group(4)))
            while key in blocks:  # frequency step repeats the same header per mode
                key = (key[0], key[1], key[2] + 1e-9)
            rows = []
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and lines[i].strip():
                try:
                    rows.append([float(x) for x in lines[i].split()])
                except ValueError:
                    break
                i += 1
            blocks[key] = np.array(rows)
            continue
        i += 1
    return dict(blocks=blocks, eigen=eigen)
