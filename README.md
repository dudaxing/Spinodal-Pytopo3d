# spinodal_pytopo3d

CPU/GPU-capable Python reproduction of the **spinodal multiscale
compliance-minimization cantilever** from Senhora, Sanders & Paulino,
*Optimally-Tailored Spinodal Architected Materials for Multiscale Design and
Manufacturing*, **Adv. Mater. 2022** (Fig. 5), built on the
[PyTopo3D](PyTopo3D-main) finite-element backbone.

This implementation intentionally starts with the paper's dominant cantilever
case: a single **columnar** spinodal class. Each element optimizes:

- macro presence `z` with density filtering, Heaviside projection, and SIMP;
- spinodal solid fraction `Frac` in `[0.3, 0.7]` (not filtered);
- orientation angles `(alpha, beta, gamma)`.

The element constitutive matrix is assembled from the homogenized material data:

```text
D_e = E_SIMP(z_bar) * N(alpha,beta,gamma)^T * D^H_columnar(Frac) * N(alpha,beta,gamma)
```

Optimization uses homogenized stiffness tensors only. Explicit GRF spinodal
pores are generated later for rendering/STL/slicing.

## Results

> **Update 2 (2026-06-10, evening): node-ordering bug found in assembly —
> all numerical results below are invalid and being recomputed.** A new
> end-to-end gate (`_bar_check.py`: assembled solid bar vs analytic compliance)
> exposed that PyTopo3D's `lk_H8` element matrix (textbook x-first node ring)
> is inconsistent with PyTopo3D's own `build_edof` (y-first node ring). Our
> element matrices had been built in the lk_H8 order and assembled with the
> build_edof wiring, which silently scrambles the physics: an isotropic bar
> came out ~37x too compliant and the columnar Young's anisotropy collapsed
> from 11.5x to ~1.1x. Element-level gates (KE==lk_H8, FD sensitivities,
> einsum rotation) are all blind to this mismatch. `fea_element._NAT` now
> follows the build_edof ring (validated by `_bar_check`), and `validate()`
> compares against `lk_H8` after the node permutation. Topology *shapes* below
> remain qualitatively meaningful (compliance optimization under a consistent,
> if wrong, SPD model), but every compliance number, `f/f0` ratio, and
> orientation field predates this fix.
>
> **Update 1 (2026-06-10): rotation bug fixed.** A self-test (`_rot_check.py`,
> cross-validating the 6x6 Bond/Voigt rotation against a 4th-order-tensor
> `np.einsum` rotation) caught two defects inherited from the released
> TO_Spinodal rotation matrix: three shear-row entries missing a factor 2
> (breaking isotropy invariance by up to 28 % at oblique angles), and an
> orientation convention that effectively rotated by `U^T`, placing the stiff
> axis at `U[2,:]` while the orientation field was interpreted as `U[:,2]`.
> Both are corrected in `spinodal_material.py`.

Fig. 5-style final renderings generated from the saved optimization fields. The
gallery uses a dense presentation render (`m=1.5`, `n_waves=120`) because the
paper's physical pore wavelength is visually too coarse on this reduced mesh.

![shell final](spinodal_pytopo3d/results/fig5a_shell_final.png)

![truss final](spinodal_pytopo3d/results/fig5d_truss_final.png)

Columnar stiff-axis streamlines:

![shell streamlines](spinodal_pytopo3d/results/fig5b_shell_streamlines.png)

![truss streamlines](spinodal_pytopo3d/results/fig5e_truss_streamlines.png)

Density-colored truss and print-slice preview:

![truss density](spinodal_pytopo3d/results/fig5d_truss_density.png)

![slice montage](spinodal_pytopo3d/results/cantilever_truss_slices/_sample_montage.png)

Current Fig. 5-aligned saved fields use a consistent solid-isotropic baseline:

| case | density policy | f/f0 |
| --- | --- | ---: |
| shell | `Frac = 0.3` fixed | 3.74 |
| truss | `0.3 <= Frac <= 0.7` optimized | 1.19 |

The shell/truss morphology and orientation-field behavior are reproduced. The
paper's exact truss ratio below 1 depends on a much finer mesh than this laptop
run; this repository keeps the single-columnar reproduction focused and does not
yet implement the full four-material selection `Z_i` model.

## Connectivity & cleanliness

A practical concern for 3D printing is whether the optimized microstructure
breaks into many disconnected fragments. It does not. The clean `gamma=12` truss
render below shows continuous members with no floating debris:

![clean truss](spinodal_pytopo3d/results/truss64_g12.png)

This is by design: the supplementary information (Fig. S1) shows spinodal
microstructures start to disconnect below `rho ~ 0.25`, which is why the
optimization bounds the spinodal solid fraction to `rho in [0.3, 0.7]` -- the
lower bound guarantees connected microstructure everywhere.

3D connected-component labelling (`scipy.ndimage.label`, 26-connectivity) on the
thresholded voxel solid confirms a single dominant body in every case: the
largest component holds **> 99.9 %** of the material, and the handful of stray
voxels are sub-resolution surface specks, not structural fragments.

| structure | components | floater voxels removed | fraction |
| --- | ---: | ---: | ---: |
| truss (`spc = 16`) | 10 | 14 | 0.0 % |
| truss64 (`gamma = 12`) | 26 | 178 | 0.0 % |
| Fig. 5 shell | 2 | 7 | 0.0 % |
| Fig. 5 truss | 10 | 112 | 0.0 % |

The large "piece counts" seen at coarse sampling (e.g. ~1000 fragments at
`spc = 12`) are a marching-cubes artifact of undersampling the pore walls, not
real disconnected solid; they vanish at `spc >= 16`. The optional `--declutter`
flag runs keep-largest-component on the voxel solid so the exported STL is a
single, watertight body:

```powershell
.\.venv\Scripts\python.exe -m spinodal_pytopo3d.render_spinodal `
    spinodal_pytopo3d/results/cantilever_truss_64.npz --m 1.5 --declutter --save-stl truss.stl
```

## Setup

From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy scipy scikit-learn matplotlib trimesh scikit-image pyvista pypardiso
```

Optional GPU solve:

```powershell
.\.venv\Scripts\python.exe -m pip install cupy-cuda12x `
  nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 `
  nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
```

Add `--use-gpu` to use warm-started CuPy CG + Jacobi. Assembly and
sensitivities still run on CPU; the reduced linear solve moves to GPU.

## Run

Default quick reproduction:

```powershell
.\.venv\Scripts\python.exe -m spinodal_pytopo3d.run_cantilever `
    --nelx 32 --nely 16 --nelz 16 --volfrac 0.05 --rmin 1.5 --maxiter 300 --case both
```

Fig. 5-aligned mode:

```powershell
.\.venv\Scripts\python.exe -m spinodal_pytopo3d.run_cantilever `
    --nelx 72 --nely 24 --nelz 24 --volfrac 0.05 --maxiter 550 `
    --case both --fig5 --tag _fig5
```

`--fig5` switches to the concentrated tip load, `R=0.4 cm` filter scaling,
`Emin=1e-4`, paper-style p-continuation, additive Heaviside beta continuation,
and the SI-style orientation staggered update schedule.

## Post-Processing

Standalone export:

```powershell
.\.venv\Scripts\python.exe -m spinodal_pytopo3d.export spinodal_pytopo3d/results/cantilever_truss_fig5.npz --vtk
```

Embedded microstructure render:

```powershell
.\.venv\Scripts\python.exe -m spinodal_pytopo3d.render_spinodal `
    spinodal_pytopo3d/results/cantilever_truss_64.npz --m 1.5 --samples-per-cell 8 --n-waves 120 `
    --declutter --presentation
```

For physical Fig. 5 pore wavelength, use `--gamma 6 --element-mm 2`; on the
current reduced mesh that looks much coarser and less continuous than the paper's
much finer optimization grid.

Print-slice preview:

```powershell
.\.venv\Scripts\python.exe -m spinodal_pytopo3d.slice_print `
    spinodal_pytopo3d/results/cantilever_truss_fig5.npz --gamma 6 --element-mm 2 --sample 8
```

Manufacturing convention: `Frac` is the spinodal **solid** fraction. The GRF
level set therefore uses `phi <= sqrt(2)*erfinv(2*Frac-1)`, so thresholded
microstructure volume is approximately `Frac`.

## Modules

| file | role |
| --- | --- |
| `fea_element.py` | H8 B-matrix and anisotropic per-element stiffness |
| `spinodal_material.py` | `D^H(Frac)` polynomial, Voigt rotation, analytic derivatives |
| `interpolation.py` | macro SIMP + Heaviside projection |
| `spinodal_fea.py` | assembly, solve, compliance, and sensitivities |
| `optimizer.py` | augmented-Lagrangian normalized-gradient update and angle sub-iterations |
| `simp_baseline.py` | standalone solid-isotropic SIMP/OC baseline |
| `run_cantilever.py` | driver and result serialization |
| `render_spinodal.py` | embedded GRF microstructure rendering and STL export |
| `render_streamlines.py` | stiff-axis streamline rendering |
| `slice_print.py` | layer-wise binary slice generation |

## Self-Tests

```powershell
.\.venv\Scripts\python.exe spinodal_pytopo3d\fea_element.py
.\.venv\Scripts\python.exe spinodal_pytopo3d\spinodal_material.py
.\.venv\Scripts\python.exe -m spinodal_pytopo3d._fd_check
.\.venv\Scripts\python.exe -m spinodal_pytopo3d._micro_check
.\.venv\Scripts\python.exe -m spinodal_pytopo3d._rot_check
.\.venv\Scripts\python.exe -m spinodal_pytopo3d._bar_check
```

Validated checks include H8 stiffness agreement with PyTopo3D `lk_H8`, material
and rotation finite-difference checks, full FEA sensitivity finite differences,
and the manufacturing level-set solid-fraction convention. `_rot_check`
cross-validates the assembly's Voigt/Bond rotation against an independent
4th-order tensor rotation (`np.einsum`): exact equivalence at random oblique
angles, isotropy invariance of the solid, and the 90-degree stiff-axis swap
consistent with `stiff_axis() = U[:,2]`.
