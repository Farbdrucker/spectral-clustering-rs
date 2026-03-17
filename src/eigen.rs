//! Eigenpair solvers for the graph Laplacian.
//!
//! Three solvers are available, chosen via [`EigenMethod`]:
//!
//! | Method          | Cost     | Quality  | Best for |
//! |-----------------|----------|----------|----------|
//! | `Full`          | O(N³)    | Exact    | N ≲ 500  |
//! | `PowerIter`     | O(N²·k·t)| ~Exact   | Any N, small k |
//! | `Lanczos`       | O(N²·q)  | ~Exact   | Any N, q ≪ N |
//!
//! All solvers return the k **smallest** eigenpairs sorted ascending.
//!
//! ### Why not just use Full always?
//!
//! For an image downscaled to max-side 64 → N = 64² = 4096.
//! Full symmetric_eigen allocates a 4096² matrix and runs O(N³) ≈ 68 billion
//! ops.  PowerIter and Lanczos converge with far fewer matrix-vector products.

use nalgebra::{DMatrix, DVector};

// ── Method selector ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum EigenMethod {
    /// Full dense symmetric eigendecomposition via nalgebra.
    /// Exact but O(N³) — only practical for N ≲ 500.
    Full,
    /// Power iteration with deflation.
    /// Finds the k *largest* eigenpairs of (2I − L), which correspond to the
    /// k *smallest* eigenpairs of L (since λ_L = 2 − λ_shifted).
    /// Converges to well-separated eigenvalues quickly; may be slow if the
    /// first few eigenvalues are clustered.
    PowerIter,
    /// Lanczos algorithm.
    /// Builds a small q×q tridiagonal matrix T via q matrix-vector products,
    /// then solves the tiny eigenproblem of T exactly.  Delivers k << q
    /// Ritz pairs that approximate the extreme eigenpairs of L well.
    /// Substantially faster than Full for large N and small k.
    Lanczos,
}

// ── Shared type alias ─────────────────────────────────────────────────────────

pub type EigenPairs = Vec<(f64, DVector<f64>)>;

// ── Full ─────────────────────────────────────────────────────────────────────

/// Exact symmetric eigendecomposition; returns k smallest pairs sorted ascending.
pub fn eigenpairs_full(l: &DMatrix<f64>, k: usize) -> EigenPairs {
    let eigen = l.clone().symmetric_eigen();
    let mut pairs: Vec<(f64, DVector<f64>)> = (0..eigen.eigenvalues.len())
        .map(|i| (eigen.eigenvalues[i], eigen.eigenvectors.column(i).into_owned()))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pairs.truncate(k);
    pairs
}

// ── Power Iteration ───────────────────────────────────────────────────────────

/// Power iteration with Gram–Schmidt deflation.
///
/// To find the *smallest* eigenvectors of L we instead find the *largest*
/// eigenvectors of the shifted matrix  M = σ_max·I − L, where σ_max = 2
/// (the theoretical upper bound for normalised Laplacian eigenvalues).
///
/// The relationship is: if (λ, v) is an eigenpair of L, then (2 − λ, v) is
/// an eigenpair of M.  The smallest λ of L → largest eigenvalue of M.
///
/// Parameters
/// ----------
/// `max_iter` — maximum iterations per eigenvector (default 500)
/// `tol`      — convergence threshold on ‖v_new − v_old‖ (default 1e-7)
pub fn eigenpairs_power(
    l: &DMatrix<f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
) -> EigenPairs {
    let n = l.nrows();

    // Build the shifted matrix M = 2I - L  once
    let mut m = -l.clone();
    for i in 0..n {
        m[(i, i)] += 2.0;
    }

    let mut found: Vec<(f64, DVector<f64>)> = Vec::with_capacity(k);

    for _ev_idx in 0..k {
        // Start from a non-trivial vector (alternating ±1 avoids alignment with
        // the all-ones trivial eigenvector of L).
        let mut v = DVector::from_fn(n, |i, _| if i % 2 == 0 { 1.0 } else { -1.0 });
        normalize(&mut v);

        for _iter in 0..max_iter {
            // Deflate: remove components along already-found eigenvectors
            for (_, u) in &found {
                let proj = u.dot(&v);
                v -= proj * u;
            }

            let v_old = v.clone();

            // Power step: v ← M·v
            v = &m * &v;

            // Re-deflate after multiplication for numerical stability
            for (_, u) in &found {
                let proj = u.dot(&v);
                v -= proj * u;
            }

            normalize(&mut v);

            let diff = (&v - &v_old).norm().min((&v + &v_old).norm()); // sign-invariant
            if diff < tol {
                break;
            }
        }

        // Rayleigh quotient: λ_L = 2 - λ_M = 2 - v^T M v
        // (since v^T M v = v^T (2I-L) v = 2 - v^T L v)
        let lambda_m = v.dot(&(&m * &v));
        let lambda_l = 2.0 - lambda_m;

        found.push((lambda_l, v));
    }

    // Sort by eigenvalue ascending (power iter finds smallest-λ_L first, but
    // degenerate/clustered cases can disorder them slightly)
    found.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    found
}

// ── Lanczos ───────────────────────────────────────────────────────────────────

/// Lanczos algorithm for the k smallest eigenpairs of a symmetric matrix L.
///
/// ### Algorithm summary
///
/// 1. Start with a random unit vector v₀.
/// 2. Build a Krylov subspace of dimension q by repeatedly multiplying by L
///    and orthogonalising (full re-orthogonalisation for numerical stability).
/// 3. The projected matrix T (q×q tridiagonal) captures the extreme eigenvalues
///    of L very well even for q ≪ N.
/// 4. Solve the small eigenproblem of T exactly (via Full solver on q×q).
/// 5. Map the Ritz vectors back to the original N-dimensional space.
///
/// Parameters
/// ----------
/// `q` — Krylov dimension (number of Lanczos steps); must satisfy k < q ≤ N.
///       Larger q → more accurate but more expensive.  Typical: q = min(N, 3k+20).
pub fn eigenpairs_lanczos(l: &DMatrix<f64>, k: usize, q: usize) -> EigenPairs {
    let n = l.nrows();
    // q must be in (k, n]; clamp to valid range
    let q = q.max(k + 1).min(n);

    // ── Build Krylov basis V (n × q) and tridiagonal T ────────────────────────
    //
    // Lanczos recurrence:
    //   β_{j+1} v_{j+1} = L v_j − α_j v_j − β_j v_{j-1}
    //   α_j = v_j^T (L v_j)
    //
    // Store V column by column.

    let mut v_cols: Vec<DVector<f64>> = Vec::with_capacity(q);
    let mut alphas: Vec<f64> = Vec::with_capacity(q);
    let mut betas: Vec<f64> = Vec::with_capacity(q); // β[0] unused; betas[j] = β_{j+1}

    // Seed vector: alternate ±1 normalised (avoids trivial eigenvector alignment)
    let mut v_cur = DVector::from_fn(n, |i, _| if i % 2 == 0 { 1.0_f64 } else { -1.0 });
    normalize(&mut v_cur);
    v_cols.push(v_cur.clone());

    let mut v_prev = DVector::zeros(n);
    let mut beta_prev = 0.0_f64;

    for j in 0..q {
        // w = L·v_j
        let mut w = l * &v_cols[j];

        // w -= α_j · v_j
        let alpha = v_cols[j].dot(&w);
        alphas.push(alpha);
        w -= alpha * &v_cols[j];

        // w -= β_j · v_{j-1}
        if j > 0 {
            w -= beta_prev * &v_prev;
        }

        // Full re-orthogonalisation against all previous vectors (stabilises
        // the computation when eigenvalues are clustered)
        for u in &v_cols {
            let proj = u.dot(&w);
            w -= proj * u;
        }

        let beta = w.norm();
        betas.push(beta);

        if j + 1 < q {
            if beta < 1e-12 {
                // Invariant subspace found; pad with a random orthogonal vector
                let mut rand_v = DVector::from_fn(n, |i, _| {
                    // Deterministic pseudo-random using bit-mixing
                    let bits = (i.wrapping_mul(2654435761).wrapping_add(j * 1000003)) as f64;
                    (bits.sin() * 43758.5453).fract()
                });
                for u in &v_cols {
                    let proj = u.dot(&rand_v);
                    rand_v -= proj * u;
                }
                normalize(&mut rand_v);
                v_prev = v_cols[j].clone();
                beta_prev = 0.0;
                v_cols.push(rand_v);
            } else {
                v_prev = v_cols[j].clone();
                beta_prev = beta;
                let v_next = w / beta;
                v_cols.push(v_next);
            }
        }
    }

    // ── Assemble the q×q tridiagonal matrix T ─────────────────────────────────
    //
    // T is symmetric tridiagonal:
    //   T[j,j]   = alpha[j]
    //   T[j,j+1] = T[j+1,j] = beta[j]   (beta[j] = β_{j+1} from above)
    let q_actual = alphas.len(); // might be < q if early termination
    let mut t = DMatrix::<f64>::zeros(q_actual, q_actual);
    for j in 0..q_actual {
        t[(j, j)] = alphas[j];
        if j + 1 < q_actual {
            t[(j, j + 1)] = betas[j];
            t[(j + 1, j)] = betas[j];
        }
    }

    // ── Solve the small eigenproblem of T exactly ──────────────────────────────
    let mut ritz_pairs = eigenpairs_full(&t, k.min(q_actual));

    // ── Map Ritz vectors back to R^N ───────────────────────────────────────────
    //
    // If y is an eigenvector of T, then  x = V y  is the Ritz vector in R^N,
    // where V is the (n × q_actual) Krylov basis.
    let v_basis = DMatrix::from_columns(&v_cols[..q_actual]);

    for (_, y) in ritz_pairs.iter_mut() {
        let x = &v_basis * &(*y); // n-dimensional Ritz vector
        *y = x;
    }

    // Sort ascending and return
    ritz_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    ritz_pairs
}

// ── Dispatcher ────────────────────────────────────────────────────────────────

pub fn compute_eigenpairs(
    l: &DMatrix<f64>,
    k: usize,
    method: &EigenMethod,
    power_max_iter: usize,
    power_tol: f64,
    lanczos_q: usize,
) -> EigenPairs {
    match method {
        EigenMethod::Full => eigenpairs_full(l, k),
        EigenMethod::PowerIter => eigenpairs_power(l, k, power_max_iter, power_tol),
        EigenMethod::Lanczos => eigenpairs_lanczos(l, k, lanczos_q),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn normalize(v: &mut DVector<f64>) {
    let norm = v.norm();
    if norm > 1e-12 {
        *v /= norm;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple normalised Laplacian of a path graph P_n.
    /// Closed-form eigenvalues: λ_k = 1 - cos(kπ/n) for k=0,…,n-1.
    fn path_laplacian(n: usize) -> DMatrix<f64> {
        // Adjacency of path graph
        let mut w = DMatrix::<f64>::zeros(n, n);
        for i in 0..n - 1 {
            w[(i, i + 1)] = 1.0;
            w[(i + 1, i)] = 1.0;
        }
        crate::laplacian::normalised_laplacian(&w)
    }

    fn check_eigenpairs(pairs: &[(f64, DVector<f64>)], l: &DMatrix<f64>, tol: f64) {
        for (lambda, v) in pairs {
            // Residual: ‖Lv − λv‖ / ‖v‖
            let residual = (l * v - *lambda * v).norm() / v.norm().max(1e-12);
            assert!(residual < tol, "residual {residual:.2e} (λ={lambda:.6})");
        }
    }

    #[test]
    fn full_path5() {
        let l = path_laplacian(5);
        let pairs = eigenpairs_full(&l, 3);
        assert_eq!(pairs.len(), 3);
        check_eigenpairs(&pairs, &l, 1e-10);
        // Smallest eigenvalue must be ~0
        assert!(pairs[0].0.abs() < 1e-10, "λ0={}", pairs[0].0);
    }

    #[test]
    fn power_iter_path5() {
        let l = path_laplacian(5);
        let pairs = eigenpairs_power(&l, 3, 1000, 1e-9);
        check_eigenpairs(&pairs, &l, 1e-5);
        assert!(pairs[0].0.abs() < 1e-5, "λ0={}", pairs[0].0);
    }

    #[test]
    fn lanczos_path5() {
        let l = path_laplacian(5);
        let pairs = eigenpairs_lanczos(&l, 3, 5);
        check_eigenpairs(&pairs, &l, 1e-8);
        assert!(pairs[0].0.abs() < 1e-8, "λ0={}", pairs[0].0);
    }

    #[test]
    fn all_methods_agree_on_small_graph() {
        // Star graph S_6 (1 hub + 5 leaves).  Well-separated spectrum.
        // We verify: (a) Lanczos matches Full exactly, (b) Power-iter finds valid
        // eigenpairs (residual-based check), not an index-by-index comparison
        // which is fragile for degenerate/clustered eigenvalues.
        let n = 6usize;
        let mut w = DMatrix::<f64>::zeros(n, n);
        for j in 1..n { w[(0,j)] = 1.0; w[(j,0)] = 1.0; }
        let l = crate::laplacian::normalised_laplacian(&w);
        let k = 3;

        // Lanczos vs Full: should match closely
        let full_eigs: Vec<f64> = eigenpairs_full(&l, k).into_iter().map(|(v,_)| v).collect();
        let lanczos_pairs = eigenpairs_lanczos(&l, k, n);
        let mut lanczos_eigs: Vec<f64> = lanczos_pairs.iter().map(|(v,_)| v.to_owned()).collect();
        lanczos_eigs.sort_by(|a,b| a.partial_cmp(b).unwrap());

        for i in 0..k {
            assert!((full_eigs[i] - lanczos_eigs[i]).abs() < 1e-6,
                "lanczos vs full at {i}: {:.6} vs {:.6}", lanczos_eigs[i], full_eigs[i]);
        }

        // Power-iter: each returned pair must satisfy the Ritz condition ‖Lv−λv‖<0.01
        let power_pairs = eigenpairs_power(&l, k, 3000, 1e-11);
        for (lambda, v) in &power_pairs {
            let residual = (&l * v - *lambda * v).norm() / v.norm().max(1e-12);
            assert!(residual < 0.01,
                "power-iter Ritz residual {residual:.4} for λ={lambda:.6}");
            // Eigenvalue must be in [0, 2] for a normalised Laplacian
            assert!(*lambda >= -1e-6 && *lambda <= 2.0 + 1e-6,
                "power-iter eigenvalue out of range: {lambda}");
        }
    }
}
