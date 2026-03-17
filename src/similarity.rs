//! Similarity / affinity matrix construction.
//!
//! Three methods are offered, selected via [`SimMethod`]:
//!
//! | Method  | Memory  | Time    | Notes |
//! |---------|---------|---------|-------|
//! | `Dense` | O(N²)   | O(N²)   | Exact; manageable up to N≈4 000 |
//! | `Knn`   | O(N·k)  | O(N²)   | Sparse (stored dense); keep k neighbours per row |
//! | `Nystrom` | O(N·m)| O(N·m²) | Low-rank approximation via m landmark pixels |
//!
//! All methods produce a **dense** `DMatrix<f64>` that is passed to the
//! Laplacian builder.  For truly large images the caller should keep N small
//! (via `--max-side`) or switch to a sparse representation; that is left as a
//! future extension.

use nalgebra::DMatrix;
use crate::features::kernel_exponent;

// ── Method selector ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum SimMethod {
    /// Full N×N Gaussian kernel matrix  (exact, O(N²) memory).
    Dense,
    /// k-nearest-neighbour sparsification: only the k most similar
    /// neighbours of each pixel are kept; all other weights are set to zero.
    /// The resulting matrix is symmetrised.  Stored dense.
    Knn,
    /// Nyström approximation: sample `m` landmark pixels uniformly at
    /// random, compute W_nm (N×m) and W_mm (m×m), then approximate
    ///   W ≈ W_nm · W_mm⁻¹ · W_nm^T
    /// This is an O(N·m²) algorithm; choose m << N.
    Nystrom,
}

// ── Dense ─────────────────────────────────────────────────────────────────────

/// Exact N×N Gaussian affinity matrix.
///
/// S_ij = exp( -(d_color²/σ_c² + d_space²/σ_s²) )
pub fn build_dense(
    feats: &[f64],
    n: usize,
    sc2: f64,
    ss2: f64,
) -> DMatrix<f64> {
    let mut s = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let fi = &feats[i * 5..(i + 1) * 5];
            let fj = &feats[j * 5..(j + 1) * 5];
            let w = (-kernel_exponent(fi, fj, sc2, ss2)).exp();
            s[(i, j)] = w;
            s[(j, i)] = w;
        }
    }
    s
}

// ── k-NN ──────────────────────────────────────────────────────────────────────

/// Sparse affinity: for each pixel keep only its `k` most similar neighbours.
///
/// Algorithm:
///   1. Compute all pairwise weights (same cost as dense).
///   2. For each row, zero out all but the top-k entries.
///   3. Symmetrise: W_sym = max(W, W^T)  (union of neighbour graphs).
///
/// Result is stored as a dense matrix.  For large N use a true sparse format;
/// here we stay dense so the rest of the pipeline is unchanged.
pub fn build_knn(
    feats: &[f64],
    n: usize,
    sc2: f64,
    ss2: f64,
    k: usize,
) -> DMatrix<f64> {
    // Step 1: full pairwise weights
    let full = build_dense(feats, n, sc2, ss2);

    // Step 2: keep only top-k per row (excluding self, which has weight 1.0)
    let k = k.min(n - 1);
    let mut sparse = DMatrix::<f64>::zeros(n, n);

    for i in 0..n {
        // Collect (weight, j) for j != i
        let mut row: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (full[(i, j)], j))
            .collect();
        // Sort descending by weight
        row.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        // Keep top-k
        for (w, j) in row.into_iter().take(k) {
            sparse[(i, j)] = w;
        }
    }

    // Step 3: symmetrise — union (take the larger of W_ij and W_ji)
    let sym = sparse.clone();
    for i in 0..n {
        for j in i + 1..n {
            let v = sym[(i, j)].max(sym[(j, i)]);
            sparse[(i, j)] = v;
            sparse[(j, i)] = v;
        }
    }
    sparse
}

// ── Nyström ───────────────────────────────────────────────────────────────────

/// Nyström low-rank approximation of the affinity matrix.
///
/// ### How it works
///
/// Partition the pixels into `m` landmark pixels (sampled uniformly) and the
/// remaining N−m pixels.  The full N×N matrix W is approximated as:
///
///   W ≈ W_nm · W_mm⁻¹ · W_nm^T
///
/// where
///   - W_mm  (m×m) — exact weights between the m landmarks
///   - W_nm  (N×m) — exact weights from every pixel to each landmark
///
/// This costs O(N·m) evaluations instead of O(N²).
/// The approximation improves as m increases; typical values: m ≈ √N … N/10.
///
/// ### Numerical note
/// We regularise W_mm with a small diagonal shift (ε·I) to avoid singularity.
pub fn build_nystrom(
    feats: &[f64],
    n: usize,
    sc2: f64,
    ss2: f64,
    m: usize,
) -> DMatrix<f64> {
    let m = m.min(n);

    // --- Sample m landmark indices uniformly without replacement ---------------
    // Deterministic sub-sampling: take every (n/m)-th pixel.
    let step = (n / m).max(1);
    let landmarks: Vec<usize> = (0..n).step_by(step).take(m).collect();
    let m = landmarks.len(); // actual count (may be slightly less than requested)

    // --- W_nm : weights from all N pixels to the m landmarks ------------------
    let mut w_nm = DMatrix::<f64>::zeros(n, m);
    for i in 0..n {
        let fi = &feats[i * 5..(i + 1) * 5];
        for (col, &lm) in landmarks.iter().enumerate() {
            let fj = &feats[lm * 5..(lm + 1) * 5];
            w_nm[(i, col)] = (-kernel_exponent(fi, fj, sc2, ss2)).exp();
        }
    }

    // --- W_mm : weights among the m landmarks (exact, symmetric) --------------
    let mut w_mm = DMatrix::<f64>::zeros(m, m);
    for (ci, &li) in landmarks.iter().enumerate() {
        for (cj, &lj) in landmarks.iter().enumerate().skip(ci) {
            let fi = &feats[li * 5..(li + 1) * 5];
            let fj = &feats[lj * 5..(lj + 1) * 5];
            let v = (-kernel_exponent(fi, fj, sc2, ss2)).exp();
            w_mm[(ci, cj)] = v;
            w_mm[(cj, ci)] = v;
        }
    }

    // --- Regularise and invert W_mm -------------------------------------------
    // ε chosen relative to the mean diagonal of W_mm (all 1s here → ε = 1e-6)
    let eps = 1e-6_f64;
    for i in 0..m {
        w_mm[(i, i)] += eps;
    }

    // Cholesky decomposition is the stable way to invert a PSD matrix.
    // nalgebra's Cholesky may fail if the matrix is not positive-definite after
    // regularisation (very rare); fall back to the full LU decomposition.
    let w_mm_inv: DMatrix<f64> = match w_mm.clone().cholesky() {
        Some(chol) => chol.inverse(),
        None => w_mm
            .clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::<f64>::identity(m, m)),
    };

    // --- W ≈ W_nm · W_mm⁻¹ · W_nm^T  ----------------------------------------
    // Compute in two steps to avoid forming an N×N intermediate that is larger
    // than the final result:
    //   A = W_nm · W_mm⁻¹   (N×m)
    //   W_approx = A · W_nm^T  (N×N)
    let a = &w_nm * &w_mm_inv;          // N×m
    let mut w_approx = &a * w_nm.transpose(); // N×N

    // Clip small negatives (numerical noise from the approximation)
    w_approx.apply(|v| *v = v.max(0.0));

    // Enforce symmetry
    for i in 0..n {
        for j in i + 1..n {
            let avg = (w_approx[(i, j)] + w_approx[(j, i)]) / 2.0;
            w_approx[(i, j)] = avg;
            w_approx[(j, i)] = avg;
        }
    }

    w_approx
}

// ── Dispatcher ────────────────────────────────────────────────────────────────

/// Build the affinity matrix using the requested method.
pub fn build_similarity(
    feats: &[f64],
    n: usize,
    sigma_color: f64,
    sigma_space: f64,
    method: &SimMethod,
    knn_k: usize,
    nystrom_m: usize,
) -> DMatrix<f64> {
    let sc2 = sigma_color * sigma_color;
    let ss2 = sigma_space * sigma_space;
    match method {
        SimMethod::Dense   => build_dense(feats, n, sc2, ss2),
        SimMethod::Knn     => build_knn(feats, n, sc2, ss2, knn_k),
        SimMethod::Nystrom => build_nystrom(feats, n, sc2, ss2, nystrom_m),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn trivial_feats(n: usize) -> Vec<f64> {
        // n pixels uniformly spaced along [0,1] in each feature dimension
        (0..n)
            .flat_map(|i| {
                let v = i as f64 / (n - 1).max(1) as f64;
                vec![v, v, v, v, v]
            })
            .collect()
    }

    #[test]
    fn dense_diagonal_is_one() {
        let n = 5;
        let feats = trivial_feats(n);
        let s = build_dense(&feats, n, 0.1, 0.1);
        for i in 0..n {
            assert!((s[(i, i)] - 1.0).abs() < 1e-10, "diag[{i}] = {}", s[(i,i)]);
        }
    }

    #[test]
    fn dense_is_symmetric() {
        let n = 6;
        let feats = trivial_feats(n);
        let s = build_dense(&feats, n, 0.2, 0.2);
        for i in 0..n {
            for j in 0..n {
                assert!((s[(i,j)] - s[(j,i)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn knn_is_symmetric() {
        let n = 8;
        let feats = trivial_feats(n);
        let s = build_knn(&feats, n, 0.2, 0.2, 3);
        for i in 0..n {
            for j in 0..n {
                assert!((s[(i,j)] - s[(j,i)]).abs() < 1e-12,
                    "knn not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn knn_sparsity() {
        let n = 10;
        let k = 2;
        let feats = trivial_feats(n);
        let s = build_knn(&feats, n, 0.3, 0.3, k);
        // After symmetrisation each row has AT MOST 2*k non-zero off-diagonal entries
        for i in 0..n {
            let nonzero = (0..n).filter(|&j| j != i && s[(i,j)] > 0.0).count();
            assert!(nonzero <= 2 * k, "row {i} has {nonzero} neighbours (expected ≤{})", 2*k);
        }
    }

    #[test]
    fn nystrom_is_symmetric() {
        let n = 10;
        let feats = trivial_feats(n);
        let s = build_nystrom(&feats, n, 0.3, 0.3, 4);
        for i in 0..n {
            for j in 0..n {
                assert!((s[(i,j)] - s[(j,i)]).abs() < 1e-10,
                    "nystrom not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn nystrom_nonnegative() {
        let n = 8;
        let feats = trivial_feats(n);
        let s = build_nystrom(&feats, n, 0.2, 0.2, 3);
        for i in 0..n {
            for j in 0..n {
                assert!(s[(i,j)] >= 0.0, "negative entry at ({i},{j}): {}", s[(i,j)]);
            }
        }
    }
}
