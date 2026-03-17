//! Normalised graph Laplacian  L = I − D^{−½} W D^{−½}
//!
//! The normalised form is preferred because it makes eigenvalues lie in [0, 2]
//! regardless of the graph's degree distribution, which is important for the
//! convergence of iterative eigensolvers.

use nalgebra::{DMatrix, DVector};

/// Build the symmetric normalised Laplacian from an affinity matrix W.
///
///   d_i   = Σ_j W_ij                    (degree of node i)
///   D     = diag(d_0, …, d_{n-1})
///   L_sym = I − D^{−½} W D^{−½}
///
/// Isolated nodes (d_i = 0) get D^{−½}_ii = 0 so they become zero rows/cols.
pub fn normalised_laplacian(w: &DMatrix<f64>) -> DMatrix<f64> {
    let n = w.nrows();

    // Degree vector
    let d: DVector<f64> = DVector::from_fn(n, |i, _| w.row(i).sum());

    // D^{-1/2}
    let d_inv_sqrt: DVector<f64> =
        d.map(|v| if v > 1e-12 { 1.0 / v.sqrt() } else { 0.0 });

    // L = I - D^{-1/2} W D^{-1/2}
    // Written out element-wise to avoid allocating an extra N×N temp.
    let mut l = DMatrix::<f64>::identity(n, n);
    for i in 0..n {
        let di = d_inv_sqrt[i];
        for j in 0..n {
            l[(i, j)] -= di * w[(i, j)] * d_inv_sqrt[j];
        }
    }
    l
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_sums_are_zero() {
        // L_sym = I - D^{-½} W D^{-½}.
        // Row sums of L_sym are zero when W has zero diagonal (pure graph weights).
        // This is because L_sym · D^{½}·1 = 0, which means D^{½}·1 is in the
        // null-space. We verify: L · D^{½} · 1 ≈ 0.
        let n = 4usize;
        let w = DMatrix::from_fn(n, n, |i, j| {
            if i == j { 0.0 } else { 1.0 / (1.0 + (i as f64 - j as f64).abs()) }
        });
        let l = normalised_laplacian(&w);

        // Build D^{½} · 1: d_i^{½} where d_i = row-sum of W
        let d_sqrt_ones: Vec<f64> = (0..n)
            .map(|i| w.row(i).sum().sqrt())
            .collect();

        for i in 0..n {
            let entry: f64 = (0..n).map(|j| l[(i,j)] * d_sqrt_ones[j]).sum();
            assert!(entry.abs() < 1e-10,
                "row {i}: (L·D^½·1)[{i}] = {entry} ≠ 0");
        }
    }

    #[test]
    fn eigenvalues_in_range() {
        // All eigenvalues of L_sym must lie in [0, 2].
        let w = DMatrix::from_fn(5, 5, |i, j| {
            if i == j { 0.0 } else { 0.5_f64.powi((i as i32 - j as i32).abs()) }
        });
        let l = normalised_laplacian(&w);
        let eigs = l.symmetric_eigen();
        for &ev in eigs.eigenvalues.iter() {
            assert!(ev >= -1e-10 && ev <= 2.0 + 1e-10, "eigenvalue out of range: {ev}");
        }
    }

    #[test]
    fn isolated_node_handled() {
        // A graph with one isolated node (zero row/column in W).
        let mut w = DMatrix::<f64>::zeros(3, 3);
        w[(0, 1)] = 1.0;
        w[(1, 0)] = 1.0;
        // Node 2 is isolated
        let l = normalised_laplacian(&w);
        // Row 2 and col 2 of L should be zero (except the diagonal 1.0 − 0 = 1)
        // Actually: d[2]=0 => D^{-1/2}[2]=0 => L[2,j]=delta[2,j]
        for j in 0..3 {
            let expected = if j == 2 { 1.0 } else { 0.0 };
            assert!((l[(2, j)] - expected).abs() < 1e-10);
        }
    }
}
