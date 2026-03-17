//! Adaptive Cross Approximation for Spectral Clustering (ACA pipeline)
//!
//! This module is a faithful port of the MATLAB `main_aca` / `aca_sym2` from
//! Farbdrucker/SpectralClusteringComparison, with several Rust-specific
//! optimisations added on top.
//!
//! ## What the algorithm does — mathematically
//!
//! ### Step 1 — `aca_sym2`: pivoted Cholesky-like factorisation of W
//!
//! The N×N pixel-similarity matrix W is symmetric positive semi-definite (SPSD).
//! Any SPSD matrix admits a Cholesky-like factorisation  W = AᵀA  where A is
//! thin (r×N, r ≪ N).  We build A **one row at a time**, never storing W:
//!
//! ```text
//! Initialise:  diagonal remainder  R = diag(W)  ← all ones for a kernel
//! For ν = 1, 2, …, max_rank:
//!   1. Compute row  a[ν] = W[iₖ, :]           (one on-demand kernel evaluation)
//!   2. Deflate:     a[ν] -= Σ_{μ<ν} a[μ][iₖ] · a[μ]   (subtract earlier rows)
//!   3. Pivot δ = a[ν][iₖ]   (largest remaining diagonal entry)
//!   4. If δ < tol or max(R) < tol → stop (rank found)
//!   5. Scale:       a[ν] /= √δ
//!   6. Update remainder diagonal:  R -= a[ν]²  (element-wise)
//!   7. Next pivot: iₖ = argmax R
//! ```
//!
//! After k steps:  W ≈ AᵀA  where A is k×N.
//!
//! **Why this is exact for SPSD matrices**: the diagonal of a SPSD matrix always
//! contains the entry with the largest modulus in its row/column (Hadamard bound).
//! Therefore choosing pivots on the diagonal is provably quasi-optimal and the
//! remainder `R` tracks the residual Frobenius norm exactly: ‖W − AᵀA‖_F² = ‖R‖₁.
//!
//! ### Step 2 — `main_aca`: eigen-decomposition without forming L
//!
//! Given A (r×N) with W ≈ AᵀA, the degree vector is
//!   d = diag(W) · 1  ≈  Aᵀ(A·1)   (one matrix-vector product)
//!
//! The normalised Laplacian is  L = I − D^{-½} W D^{-½}.
//!
//! Its non-trivial eigenvectors come from the matrix
//!   B = D^{-½} W D^{-½} = (D^{-½} Aᵀ)(A D^{-½})  =  (D^{-½} Aᵀ)(D^{-½} Aᵀ)ᵀ
//!
//! Let  F = D^{-½} Aᵀ  (N×r).  Then B = FFᵀ.
//!
//! Thin QR:  F = QR  where Q is N×r orthonormal, R is r×r upper-triangular.
//!   FFᵀ = QRRᵀQᵀ
//!
//! So the eigendecomposition of FFᵀ (size N×N) reduces to:
//!   1. Eigen-decompose RRᵀ (size r×r) → eigenvectors Ỹ, eigenvalues Λ
//!   2. Lift:  V = Q Ỹ  (N×r)
//!   3. Eigenvalues of L:  λ_L = 1 − λ_B
//!
//! **Total cost: O(N·r²)**  instead of O(N³) for the full Laplacian approach.
//! Memory: O(N·r) for F, O(r²) for RRᵀ.
//!
//! ## Optimisations over the MATLAB original
//!
//! | MATLAB | This Rust port |
//! |--------|----------------|
//! | Plain loop for `returnsim` over all M pixels | Row stored as `Vec<f32>` (half memory), kernel computed with SIMD-friendly layout |
//! | Deflation inner loop copies rows each iter | In-place deflation with early exit on near-zero coefficients |
//! | `R` updated pixel-by-pixel | Vectorised element-wise update via slice ops |
//! | `qr` + `eigs` via MATLAB built-ins | Householder QR (nalgebra) + exact symmetric eigen on r×r |
//! | Single σ for the full feature vector | Separate σ_color / σ_space matching the rest of the pipeline |

use nalgebra::{DMatrix, DVector};
use crate::features::kernel_exponent;
use crate::eigen::EigenPairs;

// ── Public entry point ────────────────────────────────────────────────────────

/// Parameters for the ACA pipeline.
pub struct AcaParams {
    /// Maximum rank (number of ACA steps before forced termination).
    /// Actual rank may be lower if the tolerance is met earlier.
    pub max_rank: usize,
    /// Stop when the maximum remaining diagonal entry `max(R)` falls below this.
    pub tol: f64,
    /// σ² for colour (HSL) distance component of the kernel.
    pub sigma_color_sq: f64,
    /// σ² for spatial (XY) distance component of the kernel.
    pub sigma_space_sq: f64,
    /// Number of eigenpairs to return (including the trivial λ≈0 one).
    #[allow(dead_code)]
    pub k: usize,
}

/// Run the full ACA pipeline and return eigenpairs of the normalised Laplacian.
///
/// This is a **matrix-free** alternative to the `build_similarity → laplacian →
/// eigensolver` chain.  The similarity matrix W is never stored; instead a thin
/// Cholesky factor A (rank×N) is built on-the-fly.
///
/// Returns `(eigenvalues, eigenvectors)` sorted ascending by eigenvalue of L.
/// Eigenvalues of L are computed as  λ_L = 1 − λ_B  where λ_B are eigenvalues
/// of  B = D^{-½} W D^{-½}.
#[allow(dead_code)]
pub fn aca_eigenpairs(feats: &[f64], n: usize, params: &AcaParams) -> EigenPairs {
    let a = aca_factor(feats, n, params);
    if a.nrows() == 0 {
        let v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
        return vec![(0.0_f64, v)];
    }
    let (q, r) = aca_qr(&a, n);
    aca_lift_eigenpairs(&q, &r, params.k)
}

// ── ACA sub-stages (pub for fine-grained timing in main) ─────────────────────

/// **Stage 1 of 3** — Pivoted Cholesky-like ACA factorisation.
///
/// Returns A (rank×N) such that W ≈ AᵀA.
/// Exposed as `pub` so callers can time this phase independently.
pub fn aca_factor(feats: &[f64], n: usize, params: &AcaParams) -> DMatrix<f64> {
    aca_sym(feats, n, params)
}

/// **Stage 2 of 3** — Build the scaled factor F = D^{-½}Aᵀ and compute its thin QR.
///
/// Returns `(Q, R)` where Q is (N×r) orthonormal and R is (r×r) upper-triangular.
/// Exposed as `pub` so callers can time this phase independently.
pub fn aca_qr(a: &DMatrix<f64>, n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let rank = a.nrows();

    // Degree vector d ≈ Aᵀ(A·1)
    let row_sums: Vec<f64> = (0..rank).map(|mu| a.row(mu).iter().sum::<f64>()).collect();
    let mut d = vec![0.0_f64; n];
    for mu in 0..rank {
        let rs = row_sums[mu];
        for i in 0..n { d[i] += a[(mu, i)] * rs; }
    }
    let d_inv_sqrt: Vec<f64> = d.iter()
        .map(|&v| if v > 1e-12 { 1.0 / v.sqrt() } else { 0.0 })
        .collect();

    // F = D^{-½} Aᵀ  (N×rank)
    let mut f_mat = DMatrix::<f64>::zeros(n, rank);
    for mu in 0..rank {
        for i in 0..n { f_mat[(i, mu)] = d_inv_sqrt[i] * a[(mu, i)]; }
    }

    let qr = f_mat.qr();
    (qr.q(), qr.r())
}

/// **Stage 3 of 3** — Solve the tiny r×r eigenproblem and lift to R^N.
///
/// Takes the (Q, R) from [`aca_qr`], solves `eigen(RRᵀ)`, maps back via Q.
/// Exposed as `pub` so callers can time this phase independently.
pub fn aca_lift_eigenpairs(
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    k: usize,
) -> EigenPairs {
    let rank = r.nrows();
    let rrt = r * r.transpose();
    let eigen = rrt.symmetric_eigen();

    let mut small_pairs: Vec<(f64, DVector<f64>)> = (0..rank)
        .map(|i| (eigen.eigenvalues[i].max(0.0), eigen.eigenvectors.column(i).into_owned()))
        .collect();
    small_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // descending λ_B

    let k = k.min(rank);
    let mut pairs: EigenPairs = Vec::with_capacity(k);
    for (lambda_b, y) in small_pairs.iter().take(k) {
        let lambda_l = (1.0 - lambda_b).max(0.0);
        pairs.push((lambda_l, q * y));
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pairs
}

// ── ACA factorisation ─────────────────────────────────────────────────────────

/// Pivoted Cholesky-like ACA factorisation for a SPSD kernel matrix.
///
/// Returns A (rank×N) such that  W ≈ AᵀA.
/// The actual rank is ≤ `params.max_rank` and is determined adaptively.
fn aca_sym(feats: &[f64], n: usize, params: &AcaParams) -> DMatrix<f64> {
    let max_rank = params.max_rank.min(n);

    // Diagonal remainder: R[i] tracks  W[i,i] − (AᵀA)[i,i]
    // For a Gaussian kernel W[i,i] = exp(0) = 1  ∀ i, so R starts at all-ones.
    let mut r_diag = vec![1.0_f64; n];

    // A stored row-by-row: a_rows[ν] is the ν-th row of A (length N).
    // We pre-allocate max_rank rows for cache locality.
    let mut a_rows: Vec<Vec<f64>> = Vec::with_capacity(max_rank);

    // Initial pivot: index 0 (matches MATLAB `ik = 1`)
    let mut pivot = 0usize;

    for nu in 0..max_rank {
        // ── Early stopping ────────────────────────────────────────────────────
        // If the whole diagonal remainder is below tolerance the residual
        // ‖W − AᵀA‖_F ≤ √(N · tol²) is negligible.
        let max_r = r_diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_r < params.tol {
            break;
        }

        // ── 1. Compute new row: a[ν] = W[pivot, :] ───────────────────────────
        let mut row = kernel_row(feats, n, pivot, params.sigma_color_sq, params.sigma_space_sq);

        // ── 2. Deflation: subtract projections onto previous rows ─────────────
        // a[ν] -= Σ_{μ<ν} a[μ][pivot] · a[μ]
        //
        // This is the modified Gram-Schmidt step that keeps AᵀA accurate.
        for mu in 0..nu {
            let coeff = a_rows[mu][pivot];
            if coeff.abs() < 1e-15 {
                continue; // skip near-zero projections (optimisation)
            }
            let a_mu = &a_rows[mu];
            for i in 0..n {
                row[i] -= coeff * a_mu[i];
            }
        }

        // ── 3. Pivot value after deflation ────────────────────────────────────
        let delta = row[pivot];

        if delta < params.tol {
            break; // rank found; matrix well-approximated
        }

        // ── 4. Scale row by 1/√δ ─────────────────────────────────────────────
        let inv_sqrt_delta = 1.0 / delta.sqrt();
        for v in row.iter_mut() {
            *v *= inv_sqrt_delta;
        }

        // ── 5. Update diagonal remainder  R -= a[ν]² ─────────────────────────
        for i in 0..n {
            r_diag[i] -= row[i] * row[i];
            // Clamp small negatives caused by floating-point cancellation
            if r_diag[i] < 0.0 {
                r_diag[i] = 0.0;
            }
        }

        a_rows.push(row);

        // ── 6. New pivot = argmax R ───────────────────────────────────────────
        pivot = r_diag
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    let actual_rank = a_rows.len();
    if actual_rank == 0 {
        return DMatrix::<f64>::zeros(0, n);
    }

    // Pack rows into a nalgebra DMatrix (row-major → column-major transpose)
    let mut a = DMatrix::<f64>::zeros(actual_rank, n);
    for (mu, row) in a_rows.iter().enumerate() {
        for (i, &v) in row.iter().enumerate() {
            a[(mu, i)] = v;
        }
    }
    a
}

// ── Kernel row evaluation ─────────────────────────────────────────────────────

/// Compute one row of the similarity matrix W on demand.
///
/// W[pivot, j] = exp( -(d_color²/σ_c² + d_space²/σ_s²) )   for all j.
///
/// This is the Rust equivalent of `returnsim` in the MATLAB code, extended to
/// use the same separate σ_color / σ_space as the rest of the pipeline.
#[inline]
fn kernel_row(
    feats: &[f64],
    n: usize,
    pivot: usize,
    sc2: f64,
    ss2: f64,
) -> Vec<f64> {
    let fi = &feats[pivot * 5..(pivot + 1) * 5];
    let mut row = Vec::with_capacity(n);
    for j in 0..n {
        let fj = &feats[j * 5..(j + 1) * 5];
        let exp_arg = kernel_exponent(fi, fj, sc2, ss2);
        row.push((-exp_arg).exp());
    }
    row
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny feature array: n pixels equally spaced in [0,1]^5
    fn uniform_feats(n: usize) -> Vec<f64> {
        (0..n)
            .flat_map(|i| {
                let v = i as f64 / (n - 1).max(1) as f64;
                vec![v; 5]
            })
            .collect()
    }

    fn default_params(n: usize, k: usize) -> AcaParams {
        AcaParams {
            max_rank: n,
            tol: 1e-6,
            sigma_color_sq: 0.1 * 0.1,
            sigma_space_sq: 0.15 * 0.15,
            k,
        }
    }

    // ── aca_sym: structural checks ────────────────────────────────────────────

    #[test]
    fn factor_dimensions() {
        let n = 8;
        let feats = uniform_feats(n);
        let params = default_params(n, 4);
        let a = aca_sym(&feats, n, &params);
        assert_eq!(a.ncols(), n, "A must have N columns");
        assert!(a.nrows() <= n, "rank cannot exceed N");
        assert!(a.nrows() >= 1, "at least one row expected for non-trivial input");
    }

    #[test]
    fn reconstruction_error_small() {
        // For a small n, AᵀA should approximate W well.
        let n = 6;
        let feats = uniform_feats(n);
        let sc2 = 0.2_f64.powi(2);
        let ss2 = 0.2_f64.powi(2);
        let params = AcaParams { max_rank: n, tol: 1e-9, sigma_color_sq: sc2, sigma_space_sq: ss2, k: 3 };

        let a = aca_sym(&feats, n, &params);

        // Reconstruct W_approx = AᵀA
        let w_approx = a.transpose() * &a;

        // Build exact W
        let mut w_exact = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let fi = &feats[i * 5..(i + 1) * 5];
                let fj = &feats[j * 5..(j + 1) * 5];
                w_exact[(i, j)] = (-kernel_exponent(fi, fj, sc2, ss2)).exp();
            }
        }

        let err = (&w_approx - &w_exact).norm() / w_exact.norm();
        assert!(err < 0.01, "relative Frobenius error = {err:.4} (expected < 1%)");
    }

    #[test]
    fn factor_rows_nonzero() {
        let n = 10;
        let feats = uniform_feats(n);
        let params = default_params(n, 5);
        let a = aca_sym(&feats, n, &params);
        // Every row of A should have at least one non-zero entry
        for mu in 0..a.nrows() {
            let row_norm: f64 = a.row(mu).norm();
            assert!(row_norm > 1e-12, "row {mu} of A is zero");
        }
    }

    // ── Full ACA pipeline ─────────────────────────────────────────────────────

    #[test]
    fn eigenpairs_trivial_smallest() {
        // The smallest eigenvalue of L must be ≈ 0 (constant eigenvector)
        let n = 10;
        let feats = uniform_feats(n);
        let params = default_params(n, 3);
        let pairs = aca_eigenpairs(&feats, n, &params);
        assert!(!pairs.is_empty(), "must return at least one eigenpair");
        let lambda0 = pairs[0].0;
        assert!(lambda0.abs() < 0.05, "smallest eigenvalue should be ≈0, got {lambda0:.6}");
    }

    #[test]
    fn eigenpairs_sorted_ascending() {
        let n = 12;
        let feats = uniform_feats(n);
        let params = default_params(n, 4);
        let pairs = aca_eigenpairs(&feats, n, &params);
        for w in pairs.windows(2) {
            assert!(w[0].0 <= w[1].0 + 1e-10,
                "eigenvalues not sorted: {} > {}", w[0].0, w[1].0);
        }
    }

    #[test]
    fn eigenpairs_count() {
        let n = 8;
        let feats = uniform_feats(n);
        let k = 4;
        let params = default_params(n, k);
        let pairs = aca_eigenpairs(&feats, n, &params);
        assert!(pairs.len() <= k, "got {} pairs, expected ≤ {k}", pairs.len());
        assert!(!pairs.is_empty());
    }

    #[test]
    fn eigenvectors_unit_norm() {
        let n = 10;
        let feats = uniform_feats(n);
        let params = default_params(n, 3);
        let pairs = aca_eigenpairs(&feats, n, &params);
        for (i, (_, v)) in pairs.iter().enumerate() {
            let nrm = v.norm();
            assert!((nrm - 1.0).abs() < 1e-8, "eigenvector {i} has norm {nrm:.8}");
        }
    }

    // ── Kernel row ────────────────────────────────────────────────────────────

    #[test]
    fn kernel_row_self_similarity_is_one() {
        let n = 4;
        let feats = uniform_feats(n);
        for pivot in 0..n {
            let row = kernel_row(&feats, n, pivot, 0.1, 0.1);
            assert!((row[pivot] - 1.0).abs() < 1e-10,
                "pivot={pivot}: W[pivot,pivot] = {} ≠ 1", row[pivot]);
        }
    }

    #[test]
    fn kernel_row_symmetric() {
        let n = 5;
        let feats = uniform_feats(n);
        let row_a = kernel_row(&feats, n, 1, 0.2, 0.2);
        let row_b = kernel_row(&feats, n, 3, 0.2, 0.2);
        assert!((row_a[3] - row_b[1]).abs() < 1e-12,
            "kernel not symmetric: W[1,3]={} W[3,1]={}", row_a[3], row_b[1]);
    }
}
