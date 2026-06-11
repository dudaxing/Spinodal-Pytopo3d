"""
Memory-lean, vectorized density-filter builder.

Produces exactly the same (H, Hs) as PyTopo3D's `build_filter`, which loops
over elements in Python and accumulates COO triplets in Python lists. At the
paper's cantilever mesh (SI S4.1: 180x30x60 = 324k elements, R = 5 elements)
that is ~170M triplets of boxed Python scalars (>15 GB) -- infeasible on
moderate-RAM machines. Here the filter is built per integer offset stencil
(di, dj, dk) with fully vectorized index arithmetic, so peak memory is a few
numpy arrays of nnz entries.

Equivalence with PyTopo3D's builder (including its integer-based distance and
epsilon conventions) is gated by `_filter_check.py`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_FLOAT_EPSILON = 1e-9   # weight cutoff, as in pytopo3d.utils.filter
_DIST_EPSILON = 1e-12   # sqrt guard for the self-offset, idem


def build_filter_fast(nelx, nely, nelz, rmin):
    """Linear (cone) density filter H (CSR) and row sums Hs.

    Element order is PyTopo3D's Fortran convention e = ely + elx*nely +
    elz*nelx*nely; neighbor distance is the integer center-to-center distance.
    """
    nele = nelx * nely * nelz
    r = int(np.ceil(rmin))

    # stencil of integer offsets with positive weight
    offs = []
    for dk in range(-r, r + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                d = np.sqrt(max(di * di + dj * dj + dk * dk, _DIST_EPSILON))
                w = rmin - d
                if w > _FLOAT_EPSILON:
                    offs.append((di, dj, dk, w))

    # exact nnz so the triplet arrays can be preallocated once
    def n_valid(n, d):
        return max(0, n - abs(d))

    counts = [n_valid(nelx, di) * n_valid(nely, dj) * n_valid(nelz, dk)
              for di, dj, dk, _ in offs]
    nnz = int(np.sum(counts))
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    vals = np.empty(nnz, dtype=np.float64)

    pos = 0
    for (di, dj, dk, w), cnt in zip(offs, counts):
        if cnt == 0:
            continue
        ex = np.arange(max(0, -di), nelx - max(0, di), dtype=np.int64)
        ey = np.arange(max(0, -dj), nely - max(0, dj), dtype=np.int64)
        ez = np.arange(max(0, -dk), nelz - max(0, dk), dtype=np.int64)
        base = (ez[:, None, None] * (nelx * nely)
                + ex[None, :, None] * nely
                + ey[None, None, :]).ravel()
        shift = dj + di * nely + dk * nelx * nely
        rows[pos:pos + cnt] = base
        cols[pos:pos + cnt] = base + shift
        vals[pos:pos + cnt] = w
        pos += cnt
    assert pos == nnz

    H = sp.csr_matrix((vals, (rows, cols)), shape=(nele, nele))
    Hs = np.asarray(H.sum(axis=1)).ravel()
    Hs[Hs == 0] = 1.0
    return H, Hs
