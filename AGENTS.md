# spinodal_pytopo3d

CPU/GPU-capable Python reproduction of the spinodal multiscale compliance-minimization
cantilever (Senhora, Sanders & Paulino, Adv. Mater. 2022, Fig. 5), built on the
[PyTopo3D](PyTopo3D-main) finite-element backbone. It is a command-line scientific
computing tool (no web/GUI server); its outputs are `.npz` fields, matplotlib design
plots, `pyvista` renders, STL/VTK exports, and print slices written under
`spinodal_pytopo3d/results/`.

See `README.md` for the full command reference (run modes, post-processing, self-tests).
The README examples use PowerShell/`.venv\Scripts\python.exe`; on this Linux VM use
`.venv/bin/python` instead.

## Cursor Cloud specific instructions

- Python deps live in a virtualenv at `.venv` (Python 3.12). Run everything with
  `.venv/bin/python ...`. The update script (re)creates `.venv` and installs
  `requirements.txt`; creating the venv requires the system `python3.12-venv` apt
  package, which is captured in the VM snapshot.
- `pypardiso` pulls in Intel MKL and is the default linear solver; it works out of the
  box here. `SpinodalFEA` automatically falls back to `scipy.sparse.linalg.spsolve` if
  pypardiso is unavailable, and `--no-pardiso` forces the scipy path.
- GPU solve (`--use-gpu`, needs `cupy`) is optional and NOT installed; there is no GPU
  in this environment, so run CPU-only (omit `--use-gpu`).
- There is no linter and no automated test framework configured. The "tests" are the
  self-check scripts listed under `## Self-Tests` in `README.md` (e.g.
  `.venv/bin/python -m spinodal_pytopo3d._bar_check`); all print `PASS`/`[OK]`.
- Run the app from the repo root so the `sys.path` insert for `PyTopo3D-main` resolves,
  e.g. `.venv/bin/python -m spinodal_pytopo3d.run_cantilever --nelx 16 --nely 8 --nelz 8
  --maxiter 40 --case both --tag _hello`. Mesh size and `--maxiter` dominate runtime; use
  a small mesh for quick smoke runs and the README's larger meshes for real reproductions.
- `render_spinodal.py` uses `pyvista` off-screen rendering and works headless here.
  `--save-stl`/`--save-ply` are flags (no path argument); files are written next to the
  input `.npz` in `spinodal_pytopo3d/results/`.
- `results/*.stl` and `results/*.ply` are gitignored (regenerable, can exceed GitHub's
  size limit); don't commit them. Avoid committing scratch run outputs (e.g. `*_hello*`).
