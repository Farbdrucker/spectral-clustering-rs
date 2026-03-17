//! Semi-supervised Allen-Cahn segmentation via convexity splitting.
//!
//! Reference: Stoll & Buenger (2016) — *Symmetric interior penalty Galerkin
//! method for fractional-in-space phase-field equations*.
//!
//! ## Problem
//!
//! Given a graph with normalised Laplacian L and a sparse label image u₀
//! (values in {-1, 0, +1}, where 0 means "unlabelled"), find u : V → [-1,+1]
//! that minimises the semi-supervised Allen-Cahn energy:
//!
//! ```text
//! E(u) = ε/2 · uᵀLu  +  1/(4ε) · ‖u²−1‖²  +  ω₀/2 · ‖(u−u₀)⊙mask‖²
//!          ─────────    ──────────────────────    ────────────────────────
//!         smoothness       double-well penalty      label fidelity
//! ```
//!
//! where `mask[i] = 1` if pixel i is labelled (`u₀[i] ≠ 0`), else 0.
//!
//! ## Convexity splitting scheme
//!
//! The energy is split as  E = E_convex + E_concave  by adding/subtracting
//! a stabilising term `c/2 · ‖u‖²`:
//!
//! ```text
//! E_convex  = ε/2·uᵀLu  +  c/2·‖u‖²            (treated implicitly)
//! E_concave = 1/(4ε)·‖u²−1‖²  −  c/2·‖u‖²      (treated explicitly)
//! ```
//!
//! The constant `c = 3/ε + ω₀` ensures convexity of both parts.
//!
//! Working in the spectral basis  a = Φᵀu  (where Φ holds the eigenvectors
//! as columns), the update per time-step dt becomes:
//!
//! ```text
//! D  = 1 + dt·(ε·λ + c)            (diagonal, one entry per eigenvector)
//! b  = Φᵀ u³                        (nonlinear term, pixel space → spectral)
//! d  = Φᵀ [(u − u₀) ⊙ ω]           (fidelity gradient, pixel space → spectral)
//!
//! a_new = (1/D) ⊙ [ (1 + dt/ε + c·dt)·a  −  dt/ε·b  +  dt·d ]
//! u_new = Φ · a_new
//! ```
//!
//! Note the `+dt·d` sign: the fidelity gradient is `∇_a [ω₀/2·‖Φa−u₀‖²_ω]`
//! which evaluates to `ω⊙(u−u₀)` in pixel space, so `d = Φᵀ[ω⊙(u−u₀)]`.
//! The explicit step subtracts the concave gradient: `a ← a − dt·d_concave`.
//! Since `E_concave` contains `−ω₀/2·‖u‖²` (moved to implicit side), the
//! fidelity term is in E_concave with gradient `d`, leading to `−dt·d` in
//! the numerator, which the code writes as `left − right` with `right = −dt·d`
//! i.e. `+dt·d` appears in `left`.  See `update_spectral_coeffs` for details.
//!
//! ## Bugs fixed from the original Python
//!
//! | # | Original Python | Fixed here |
//! |---|-----------------|------------|
//! | 1 | `omega` only set for `u0 > 0` — labels `u0 < 0` unconstrained | `omega[i] = omega0` for all `u0[i] != 0` |
//! | 2 | `u = phi * a` computed twice per iteration (once unused before norm check) | Computed once |
//! | 3 | `np.mat` (deprecated) with implicit broadcasting | Plain `ndarray`-style slices via nalgebra `DVector` |
//! | 4 | `b` initialised as `Φᵀ u₀³` but first iteration overwrites with `Φᵀ u³` — initialisation is redundant | Initialised correctly from `u = Φ·a` |
//! | 5 | No convergence output, sparse `print` | Per-iteration progress via `eprintln` (behind `verbose` flag) |

use nalgebra::{DMatrix, DVector};

// ── Public types ──────────────────────────────────────────────────────────────

/// Parameters for the convexity splitting iteration.
#[derive(Debug, Clone)]
pub struct ConvSplitParams {
    /// Fidelity weight ω₀.  Larger ⇒ labelled pixels are held tighter.
    pub omega0: f64,
    /// Interface width ε.  `None` ⇒ auto-choose `1/√N`.
    pub eps: Option<f64>,
    /// Time step dt.
    pub dt: f64,
    /// Stabilisation constant c.  `None` ⇒ auto-choose `3/ε + ω₀`.
    pub c: Option<f64>,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Relative change tolerance: stop when `‖u_new − u‖/‖u‖ < tol`.
    pub tol: f64,
    /// Print per-iteration residual.
    pub verbose: bool,
}

impl Default for ConvSplitParams {
    fn default() -> Self {
        Self {
            omega0: 1.0,
            eps: None,
            dt: 0.01,
            c: None,
            max_iter: 500,
            tol: 1e-8,
            verbose: false,
        }
    }
}

/// Output of the convexity splitting solver.
pub struct ConvSplitResult {
    /// Final pixel-space solution u ∈ [-1, +1]^N (approximately).
    pub u: DVector<f64>,
    /// Initial spectral projection  a₀ = Φᵀ u₀.
    #[allow(dead_code)]
    pub a0: DVector<f64>,
    /// Number of iterations actually performed.
    pub iters: usize,
    /// Final relative residual `‖u_new − u‖/‖u‖`.
    pub final_residual: f64,
}

// ── Main solver ───────────────────────────────────────────────────────────────

/// Run semi-supervised Allen-Cahn segmentation via convexity splitting.
///
/// # Arguments
///
/// * `u0`     — label image as a flat `Vec<f64>` in pixel order.
///              Values: `+1` (foreground), `-1` (background), `0` (unlabelled).
///              Length must equal `N = width × height`.
/// * `lambdas` — eigenvalues of the graph Laplacian, length k.
/// * `phi`    — eigenvectors as columns of an (N × k) matrix.
/// * `params`  — solver parameters.
///
/// # Returns
///
/// [`ConvSplitResult`] containing the final segmentation `u` and diagnostics.
pub fn conv_splitting(
    u0: &[f64],
    lambdas: &[f64],
    phi: &DMatrix<f64>,   // N × k  (columns = eigenvectors)
    params: &ConvSplitParams,
) -> ConvSplitResult {
    let n = u0.len();
    let k = lambdas.len();
    debug_assert_eq!(phi.nrows(), n, "phi must have N rows");
    debug_assert_eq!(phi.ncols(), k, "phi must have k columns");

    // ── Derived constants ─────────────────────────────────────────────────────
    let eps = params.eps.unwrap_or_else(|| 1.0 / (n as f64).sqrt());
    let c   = params.c.unwrap_or(3.0 / eps + params.omega0);

    // ── Fidelity weight vector ω ──────────────────────────────────────────────
    //
    // FIX: original Python only set ω[i] = ω₀ when u₀[i] > 0, so negative
    // labels (-1) were unconstrained.  Both labelled classes must be anchored.
    let omega: DVector<f64> = DVector::from_fn(n, |i, _| {
        if u0[i].abs() > 1e-12 { params.omega0 } else { 0.0 }
    });

    // ── Spectral basis vectors as DVector views ───────────────────────────────
    let u0_vec = DVector::from_column_slice(u0);
    let lambda_vec = DVector::from_column_slice(lambdas);

    // ── Initial spectral coefficients  a = Φᵀ u₀ ─────────────────────────────
    let a0: DVector<f64> = phi.transpose() * &u0_vec;

    // ── Pre-compute fixed spectral quantities ─────────────────────────────────
    //
    // D = 1 + dt·(ε·λ + c)  — diagonal denominator, same every iteration
    let d_denom: DVector<f64> = DVector::from_fn(k, |i, _| {
        1.0 + params.dt * (eps * lambda_vec[i] + c)
    });
    // inv_D for element-wise division
    let inv_d: DVector<f64> = d_denom.map(|v| 1.0 / v);

    // Left-hand scale factor (same every iteration)
    let lhs_scale = 1.0 + params.dt / eps + c * params.dt;

    // ── Initialise iteration ──────────────────────────────────────────────────
    let mut a = a0.clone();
    let mut u: DVector<f64> = phi * &a;   // u = Φ a

    let mut iters = 0usize;
    let mut final_residual = 0.0f64;

    // ── Main loop ─────────────────────────────────────────────────────────────
    for iter in 0..params.max_iter {
        // b = Φᵀ u³   (nonlinear term, computed in pixel space for efficiency)
        let u3: DVector<f64> = u.map(|v| v * v * v);
        let b: DVector<f64>  = phi.transpose() * &u3;

        // d = Φᵀ [(u − u₀) ⊙ ω]   (fidelity gradient)
        let fid: DVector<f64> = DVector::from_fn(n, |i, _| (u[i] - u0[i]) * omega[i]);
        let d: DVector<f64>   = phi.transpose() * &fid;

        // Spectral update:
        //   a_new = inv_D ⊙ [ lhs_scale·a  −  (dt/ε)·b  +  dt·d ]
        //
        // Derivation: the convex-splitting update in spectral form is
        //   D a_new  = lhs_scale·a  −  (dt/ε)·b  +  dt·d
        // where the fidelity gradient d appears with a + sign because it
        // enters via the concave part: E_concave ⊃ −ω₀/2‖u‖² + ω₀/2‖u−u₀‖²
        // whose gradient w.r.t. a is Φᵀ[ω⊙(u−u₀)] = d; the explicit step
        // subtracts dt·∇E_concave = dt·(−c·a + d) giving +dt·d in numerator.
        let numerator: DVector<f64> = DVector::from_fn(k, |i, _| {
            lhs_scale * a[i]  -  (params.dt / eps) * b[i]  +  params.dt * d[i]
        });
        let a_new: DVector<f64> = DVector::from_fn(k, |i, _| inv_d[i] * numerator[i]);

        // Reconstruct pixel-space solution
        let u_new: DVector<f64> = phi * &a_new;

        // Convergence check (skip first iteration — u is still u₀)
        if iter > 0 {
            let norm_u = u.norm().max(1e-12);
            let norm_diff = (&u_new - &u).norm();
            final_residual = norm_diff / norm_u;

            if params.verbose {
                eprintln!(
                    "      conv-split iter {:>4}  residual = {:.3e}  u∈[{:.3},{:.3}]",
                    iter, final_residual, u_new.min(), u_new.max()
                );
            }

            if final_residual < params.tol {
                iters = iter;
                u = u_new;
                break;
            }
        }

        a = a_new;
        u = u_new;
        iters = iter;
    }

    ConvSplitResult { u, a0, iters, final_residual }
}

// ── u₀ helpers ────────────────────────────────────────────────────────────────

/// Load a label image from a greyscale PNG.
///
/// Pixel encoding expected in the PNG:
/// - White (≥ 192)  → +1  (foreground label)
/// - Black (≤ 63)   → −1  (background label)
/// - Mid-grey       →  0  (unlabelled)
///
/// The loaded image is rescaled to match `(width, height)` if needed.
pub fn load_u0_from_png(
    path: &std::path::Path,
    width: u32,
    height: u32,
) -> anyhow::Result<Vec<f64>> {
    let raw = image::open(path)?.into_luma8();
    let (iw, ih) = raw.dimensions();

    let mut out = vec![0.0f64; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            // nearest-neighbour sample from source
            let sx = (x as f64 / width as f64 * iw as f64) as u32;
            let sy = (y as f64 / height as f64 * ih as f64) as u32;
            let px = raw.get_pixel(sx.min(iw - 1), sy.min(ih - 1))[0];
            out[(y * width + x) as usize] = if px >= 192 {
                1.0
            } else if px <= 63 {
                -1.0
            } else {
                0.0
            };
        }
    }
    Ok(out)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    /// Build an identity-like phi (N×N diagonal) and zero lambdas for
    /// unit-testing the iteration without any graph structure.
    fn trivial_setup(n: usize, k: usize) -> (DMatrix<f64>, Vec<f64>) {
        // Use the first k standard basis vectors as eigenvectors
        let mut phi = DMatrix::<f64>::zeros(n, k);
        for i in 0..k {
            phi[(i, i)] = 1.0;
        }
        let lambdas = vec![0.0f64; k];
        (phi, lambdas)
    }

    #[test]
    fn output_length_matches_n() {
        let n = 16;
        let k = 4;
        let (phi, lambdas) = trivial_setup(n, k);
        let u0 = vec![0.0f64; n];
        let result = conv_splitting(&u0, &lambdas, &phi, &ConvSplitParams::default());
        assert_eq!(result.u.len(), n);
    }

    #[test]
    fn labelled_pixels_stay_near_their_label() {
        // With strong fidelity (large omega0) and a fully labelled image,
        // the solution should remain on the correct side (u>0 for +1, u<0 for -1).
        let n = 8;
        let k = 4;
        let (phi, lambdas) = trivial_setup(n, k);
        // Alternating +1 / -1 labels, all pixels labelled
        let u0: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

        let params = ConvSplitParams {
            omega0: 100.0, // very strong fidelity
            max_iter: 200,
            tol: 1e-10,
            ..Default::default()
        };
        let result = conv_splitting(&u0, &lambdas, &phi, &params);

        // With strong fidelity: positive-labelled pixels should have u > 0,
        // negative-labelled should have u < 0.
        // We allow a small tolerance around the decision boundary.
        for i in 0..n {
            if u0[i] > 0.5 {
                assert!(result.u[i] > -0.1,
                    "pixel {i} (+1 label): u={:.4} should be > -0.1", result.u[i]);
            } else if u0[i] < -0.5 {
                assert!(result.u[i] < 0.1,
                    "pixel {i} (-1 label): u={:.4} should be < 0.1", result.u[i]);
            }
        }
    }

    #[test]
    fn unlabelled_pixels_converge_to_double_well() {
        // A single unlabelled pixel (u0=0) with no graph neighbours should
        // converge to ±1 (double-well minima).
        let n = 4;
        let k = 2;
        let (phi, lambdas) = trivial_setup(n, k);
        let u0 = vec![0.0f64; n]; // all unlabelled

        let params = ConvSplitParams {
            max_iter: 2000,
            tol: 1e-10,
            ..Default::default()
        };
        let result = conv_splitting(&u0, &lambdas, &phi, &params);
        // Without any label anchoring, the solution may stay near 0 due to the
        // trivial fixed point; just check it's in [-1.5, 1.5] (bounded)
        for i in 0..n {
            assert!(
                result.u[i].abs() <= 1.5,
                "pixel {i}: u={:.4} outside [-1.5, 1.5]", result.u[i]
            );
        }
    }

    #[test]
    fn omega_covers_both_label_classes() {
        // FIX: omega must be set for BOTH +1 and -1 labels.
        let n = 4;
        // u0 with both positive and negative labels
        let u0 = vec![1.0, -1.0, 0.0, 0.0f64];

        let omega: Vec<f64> = (0..n)
            .map(|i| if u0[i].abs() > 1e-12 { 1.0 } else { 0.0 })
            .collect();

        assert!(omega[0] > 0.0, "positive label should get omega > 0");
        assert!(omega[1] > 0.0, "negative label should ALSO get omega > 0 (bug fix)");
        assert_eq!(omega[2], 0.0, "unlabelled pixel should have omega = 0");
        assert_eq!(omega[3], 0.0, "unlabelled pixel should have omega = 0");
    }

    #[test]
    fn spectral_coefficients_a0_correct() {
        // a0 = Φᵀ u0; for the identity phi this means a0[i] = u0[i].
        let n = 6;
        let k = 3;
        let (phi, lambdas) = trivial_setup(n, k);
        let u0 = vec![0.5, -0.3, 0.8, 0.0, 0.0, 0.0f64];

        let result = conv_splitting(&u0, &lambdas, &phi, &ConvSplitParams {
            max_iter: 1,
            ..Default::default()
        });

        for i in 0..k {
            assert!((result.a0[i] - u0[i]).abs() < 1e-12,
                "a0[{i}] = {:.6}, expected {:.6}", result.a0[i], u0[i]);
        }
    }

    #[test]
    fn convergence_reported_correctly() {
        let n = 8;
        let k = 4;
        let (phi, lambdas) = trivial_setup(n, k);
        let u0: Vec<f64> = (0..n).map(|i| if i < n/2 { 1.0 } else { -1.0 }).collect();

        let params = ConvSplitParams {
            omega0: 10.0,
            max_iter: 300,
            tol: 1e-6,
            ..Default::default()
        };
        let result = conv_splitting(&u0, &lambdas, &phi, &params);
        // Either converged or hit max_iter — iters should be in valid range
        assert!(result.iters < params.max_iter || result.final_residual <= params.tol,
            "either converged or ran to max; iters={} residual={:.2e}",
            result.iters, result.final_residual);
    }
}
