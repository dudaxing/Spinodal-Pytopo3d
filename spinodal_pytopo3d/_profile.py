"""Quick profiler + correctness check for analyze() after the (1)+(2) speedups."""
import cProfile
import io
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PyTopo3D-main"))
from pytopo3d.utils.assembly import build_force_vector, build_supports  # noqa
from pytopo3d.utils.filter import build_filter  # noqa

from spinodal_pytopo3d.spinodal_fea import SpinodalFEA, SpinodalParams

nelx, nely, nelz = 32, 16, 16
nele = nelx * nely * nelz
ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
print(f"mesh {nelx}x{nely}x{nelz} = {nele} elems, ndof={ndof}")

F = build_force_vector(nelx, nely, nelz, ndof)
free, _ = build_supports(nelx, nely, nelz, ndof)
H, Hs = build_filter(nelx, nely, nelz, 1.6)

rng = np.random.default_rng(0)
z = rng.uniform(0.3, 0.7, nele)
Frac = rng.uniform(0.3, 0.7, nele)
a = rng.uniform(-1, 1, nele); b = rng.uniform(-1, 1, nele); g = rng.uniform(-1, 1, nele)
p = SpinodalParams(penal=3.0, beta=3.0)

# ---- correctness: persistent PyPardiso path vs SciPy path ----
fea_ps = SpinodalFEA(nelx, nely, nelz, F, free, H, Hs, use_pardiso=True)
fea_sp = SpinodalFEA(nelx, nely, nelz, F, free, H, Hs, use_pardiso=False)
print("solver:", fea_ps.solver_name)
o1 = fea_ps.analyze(z, Frac, a, b, g, p)
o2 = fea_sp.analyze(z, Frac, a, b, g, p)
print(f"c (pardiso) = {o1['c']:.8e}")
print(f"c (scipy)   = {o2['c']:.8e}")
print(f"max|U_ps - U_sp| = {np.max(np.abs(o1['U']-o2['U'])):.2e}")
print(f"F^T U (pardiso)  = {float(F @ o1['U']):.8e}  (== c?)")

# perturb and re-solve to confirm persistent solver does NOT return stale results
z2 = z.copy(); z2[10] += 0.1
o3 = fea_ps.analyze(z2, Frac, a, b, g, p)
o3s = fea_sp.analyze(z2, Frac, a, b, g, p)
print(f"after perturb: c_ps={o3['c']:.6e}  c_sp={o3s['c']:.6e}  "
      f"rel diff={abs(o3['c']-o3s['c'])/abs(o3s['c']):.2e}")

# ---- speed ----
fea_ps.analyze(z, Frac, a, b, g, p)  # warmup
N = 10
t0 = time.perf_counter()
for _ in range(N):
    fea_ps.analyze(z, Frac, a, b, g, p)
print(f"\nfull analyze() [pardiso]: {(time.perf_counter()-t0)/N*1000:.1f} ms/call")

t0 = time.perf_counter()
for _ in range(N):
    fea_sp.analyze(z, Frac, a, b, g, p)
print(f"full analyze() [scipy]  : {(time.perf_counter()-t0)/N*1000:.1f} ms/call")

print("\n---- cProfile of 15 analyze() [pardiso] (top tottime) ----")
pr = cProfile.Profile(); pr.enable()
for _ in range(15):
    fea_ps.analyze(z, Frac, a, b, g, p)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(10)
print("\n".join(s.getvalue().splitlines()[:20]))
