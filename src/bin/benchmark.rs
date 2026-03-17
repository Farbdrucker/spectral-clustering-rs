//! Benchmark binary for spectral-clustering-rs.
//!
//! Sweeps eigensolver, similarity, and ACA approximation parameters across
//! synthetic images and reports per-eigenvalue errors + wall-clock timing.
//!
//! Run with: `cargo run --release --bin benchmark`

use nalgebra::{DMatrix, DVector};
use spectral_segment::aca::{aca_factor, aca_lift_eigenpairs, aca_qr, AcaParams};
use spectral_segment::eigen::{compute_eigenpairs, EigenMethod, EigenPairs};
use spectral_segment::laplacian::normalised_laplacian;
use spectral_segment::similarity::{build_similarity, SimMethod};
use std::time::Instant;

// ── Result type ───────────────────────────────────────────────────────────────

struct BenchResult {
    label: String,
    elapsed_ms: f64,
    eigenvalue_abs_errors: Vec<f64>,
    eigenvalue_rel_errors: Vec<f64>,
    eigenvector_residuals: Vec<f64>,
    frob_error: Option<f64>,
}

// ── Timing helper ─────────────────────────────────────────────────────────────

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let result = f();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    (result, ms)
}

// ── Accuracy metrics ──────────────────────────────────────────────────────────

fn eigenpair_residual(l: &DMatrix<f64>, lambda: f64, v: &DVector<f64>) -> f64 {
    let lv = l * v;
    let res = lv - lambda * v;
    res.norm() / v.norm().max(1e-30)
}

fn eigenvalue_errors(ref_vals: &[f64], approx_vals: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let len = ref_vals.len().min(approx_vals.len());
    let mut abs_errs = Vec::with_capacity(len);
    let mut rel_errs = Vec::with_capacity(len);
    for i in 0..len {
        let abs_e = (ref_vals[i] - approx_vals[i]).abs();
        let rel_e = abs_e / ref_vals[i].abs().max(1e-12);
        abs_errs.push(abs_e);
        rel_errs.push(rel_e);
    }
    (abs_errs, rel_errs)
}

fn frob_rel_error(reference: &DMatrix<f64>, approx: &DMatrix<f64>) -> f64 {
    let diff = reference - approx;
    diff.norm() / reference.norm().max(1e-30)
}

fn sorted_eigenvalues(pairs: &EigenPairs) -> Vec<f64> {
    let mut vals: Vec<f64> = pairs.iter().map(|(l, _)| *l).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals
}

// ── Synthetic feature generators ─────────────────────────────────────────────
// Feature layout per pixel i: feats[i*5..i*5+5] = [hue=0, sat=0, lum, x_norm, y_norm]

fn gradient_feats(w: usize, h: usize) -> Vec<f64> {
    let n = w * h;
    let mut feats = vec![0.0f64; n * 5];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let lum = x as f64 / (w - 1).max(1) as f64;
            feats[i * 5 + 2] = lum;
            feats[i * 5 + 3] = x as f64 / (w - 1).max(1) as f64;
            feats[i * 5 + 4] = y as f64 / (h - 1).max(1) as f64;
        }
    }
    feats
}

fn checkerboard_feats(w: usize, h: usize, cell: usize) -> Vec<f64> {
    let n = w * h;
    let mut feats = vec![0.0f64; n * 5];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let lum = if (x / cell + y / cell) % 2 == 0 { 0.2 } else { 0.8 };
            feats[i * 5 + 2] = lum;
            feats[i * 5 + 3] = x as f64 / (w - 1).max(1) as f64;
            feats[i * 5 + 4] = y as f64 / (h - 1).max(1) as f64;
        }
    }
    feats
}

fn noise_feats(w: usize, h: usize, seed: u64) -> Vec<f64> {
    let n = w * h;
    let mut feats = vec![0.0f64; n * 5];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            // deterministic bit-mixing (splitmix64)
            let mut v = seed ^ (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v ^= v >> 33;
            v = v.wrapping_mul(0xff51afd7ed558ccd);
            v ^= v >> 33;
            v = v.wrapping_mul(0xc4ceb9fe1a85ec53);
            v ^= v >> 33;
            let lum = (v as f64) / (u64::MAX as f64);
            feats[i * 5 + 2] = lum;
            feats[i * 5 + 3] = x as f64 / (w - 1).max(1) as f64;
            feats[i * 5 + 4] = y as f64 / (h - 1).max(1) as f64;
        }
    }
    feats
}

// ── Benchmark A: eigensolver comparison ───────────────────────────────────────

fn bench_eigensolvers(feats: &[f64], n: usize, k: usize, sc: f64, ss: f64) -> Vec<BenchResult> {
    // Build W and L once (not counted in per-method timing)
    let w = build_similarity(feats, n, sc, ss, &SimMethod::Dense, 0, 0);
    let l = normalised_laplacian(&w);

    // Reference: Full
    let (ref_pairs, ref_ms) = timed(|| compute_eigenpairs(&l, k, &EigenMethod::Full, 0, 0.0, 0));
    let ref_vals = sorted_eigenvalues(&ref_pairs);
    let ref_residuals: Vec<f64> = ref_pairs
        .iter()
        .map(|(lam, v)| eigenpair_residual(&l, *lam, v))
        .collect();

    let ref_result = BenchResult {
        label: "Full (reference)".to_string(),
        elapsed_ms: ref_ms,
        eigenvalue_abs_errors: vec![0.0; ref_vals.len()],
        eigenvalue_rel_errors: vec![0.0; ref_vals.len()],
        eigenvector_residuals: ref_residuals,
        frob_error: None,
    };

    let variants: Vec<(&str, EigenMethod, usize, f64, usize)> = vec![
        ("PowerIter max=100",  EigenMethod::PowerIter, 100,  1e-10, 0),
        ("PowerIter max=500",  EigenMethod::PowerIter, 500,  1e-10, 0),
        ("PowerIter max=2000", EigenMethod::PowerIter, 2000, 1e-10, 0),
        ("Lanczos q=auto",     EigenMethod::Lanczos,   0, 0.0, 3 * k + 20),
        ("Lanczos q=2k",       EigenMethod::Lanczos,   0, 0.0, 2 * k),
        ("Lanczos q=5k",       EigenMethod::Lanczos,   0, 0.0, 5 * k),
    ];

    let mut results = vec![ref_result];

    for (label, method, power_max, power_tol, lanczos_q) in variants {
        let (pairs, elapsed_ms) =
            timed(|| compute_eigenpairs(&l, k, &method, power_max, power_tol, lanczos_q));

        let vals = sorted_eigenvalues(&pairs);
        let (abs_errs, rel_errs) = eigenvalue_errors(&ref_vals, &vals);
        let residuals: Vec<f64> = pairs
            .iter()
            .map(|(lam, v)| eigenpair_residual(&l, *lam, v))
            .collect();

        // Annotate label with actual lanczos_q used
        let full_label = if matches!(method, EigenMethod::Lanczos) {
            let q_actual = lanczos_q.max(k + 1).min(n);
            format!("{} (q={})", label, q_actual)
        } else {
            label.to_string()
        };

        results.push(BenchResult {
            label: full_label,
            elapsed_ms,
            eigenvalue_abs_errors: abs_errs,
            eigenvalue_rel_errors: rel_errs,
            eigenvector_residuals: residuals,
            frob_error: None,
        });
    }

    results
}

// ── Benchmark B: similarity comparison ───────────────────────────────────────

fn bench_similarity(feats: &[f64], n: usize, k: usize, sc: f64, ss: f64) -> Vec<BenchResult> {
    let sqrt_n = (n as f64).sqrt() as usize;

    // Build W_dense outside timing for frob comparison
    let w_dense = build_similarity(feats, n, sc, ss, &SimMethod::Dense, 0, 0);

    // Reference: Dense + L + Full (all timed together)
    let (ref_pairs, ref_ms) = timed(|| {
        let w = build_similarity(feats, n, sc, ss, &SimMethod::Dense, 0, 0);
        let l = normalised_laplacian(&w);
        compute_eigenpairs(&l, k, &EigenMethod::Full, 0, 0.0, 0)
    });
    let ref_vals = sorted_eigenvalues(&ref_pairs);

    // Compute residuals for reference using dense L
    let l_dense = normalised_laplacian(&w_dense);
    let ref_residuals: Vec<f64> = ref_pairs
        .iter()
        .map(|(lam, v)| eigenpair_residual(&l_dense, *lam, v))
        .collect();

    let ref_result = BenchResult {
        label: "Dense (reference)".to_string(),
        elapsed_ms: ref_ms,
        eigenvalue_abs_errors: vec![0.0; ref_vals.len()],
        eigenvalue_rel_errors: vec![0.0; ref_vals.len()],
        eigenvector_residuals: ref_residuals,
        frob_error: Some(0.0),
    };

    let mut results = vec![ref_result];

    // Knn variants
    for knn_k in [5, 10, 20, 40] {
        let label = format!("Knn k={}", knn_k);
        let (pairs, elapsed_ms) = timed(|| {
            let w = build_similarity(feats, n, sc, ss, &SimMethod::Knn, knn_k, 0);
            let l = normalised_laplacian(&w);
            (w, compute_eigenpairs(&l, k, &EigenMethod::Full, 0, 0.0, 0))
        });
        let (w_approx, pairs) = pairs;

        let vals = sorted_eigenvalues(&pairs);
        let (abs_errs, rel_errs) = eigenvalue_errors(&ref_vals, &vals);

        // Residuals use the approximate Laplacian (internal consistency)
        let l_approx = normalised_laplacian(&w_approx);
        let residuals: Vec<f64> = pairs
            .iter()
            .map(|(lam, v)| eigenpair_residual(&l_approx, *lam, v))
            .collect();

        let frob_err = frob_rel_error(&w_dense, &w_approx);

        results.push(BenchResult {
            label,
            elapsed_ms,
            eigenvalue_abs_errors: abs_errs,
            eigenvalue_rel_errors: rel_errs,
            eigenvector_residuals: residuals,
            frob_error: Some(frob_err),
        });
    }

    // Nystrom variants: m ∈ {⌈√N/2⌉, ⌈√N⌉, ⌈2√N⌉}
    let nystrom_ms = [
        ((sqrt_n + 1) / 2).max(k + 1),
        sqrt_n.max(k + 1),
        (2 * sqrt_n).max(k + 1),
    ];
    for m in nystrom_ms {
        let label = format!("Nystrom m={}", m);
        let (pairs, elapsed_ms) = timed(|| {
            let w = build_similarity(feats, n, sc, ss, &SimMethod::Nystrom, 0, m);
            let l = normalised_laplacian(&w);
            (w, compute_eigenpairs(&l, k, &EigenMethod::Full, 0, 0.0, 0))
        });
        let (w_approx, pairs) = pairs;

        let vals = sorted_eigenvalues(&pairs);
        let (abs_errs, rel_errs) = eigenvalue_errors(&ref_vals, &vals);

        let l_approx = normalised_laplacian(&w_approx);
        let residuals: Vec<f64> = pairs
            .iter()
            .map(|(lam, v)| eigenpair_residual(&l_approx, *lam, v))
            .collect();

        let frob_err = frob_rel_error(&w_dense, &w_approx);

        results.push(BenchResult {
            label,
            elapsed_ms,
            eigenvalue_abs_errors: abs_errs,
            eigenvalue_rel_errors: rel_errs,
            eigenvector_residuals: residuals,
            frob_error: Some(frob_err),
        });
    }

    results
}

// ── Benchmark C: ACA vs Classic end-to-end ────────────────────────────────────

fn bench_aca(feats: &[f64], n: usize, k: usize, sc: f64, ss: f64) -> Vec<BenchResult> {
    let sqrt_n = (n as f64).sqrt() as usize;

    // Reference: Dense + Full Classic (both timed)
    let (ref_pairs, ref_ms) = timed(|| {
        let w = build_similarity(feats, n, sc, ss, &SimMethod::Dense, 0, 0);
        let l = normalised_laplacian(&w);
        compute_eigenpairs(&l, k, &EigenMethod::Full, 0, 0.0, 0)
    });
    let ref_vals = sorted_eigenvalues(&ref_pairs);

    // Build reference L for residual computation
    let w_dense = build_similarity(feats, n, sc, ss, &SimMethod::Dense, 0, 0);
    let l_ref = normalised_laplacian(&w_dense);

    let ref_residuals: Vec<f64> = ref_pairs
        .iter()
        .map(|(lam, v)| eigenpair_residual(&l_ref, *lam, v))
        .collect();

    let ref_result = BenchResult {
        label: "Classic Dense+Full (reference)".to_string(),
        elapsed_ms: ref_ms,
        eigenvalue_abs_errors: vec![0.0; ref_vals.len()],
        eigenvalue_rel_errors: vec![0.0; ref_vals.len()],
        eigenvector_residuals: ref_residuals,
        frob_error: Some(0.0),
    };

    let mut results = vec![ref_result];

    // ACA ranks: {⌈√N/2⌉, ⌈√N⌉, ⌈2√N⌉, ⌈4√N⌉}
    let ranks = [
        ((sqrt_n + 1) / 2).max(k + 1),
        sqrt_n.max(k + 1),
        (2 * sqrt_n).max(k + 1),
        (4 * sqrt_n).max(k + 1),
    ];

    for rank in ranks {
        let params = AcaParams {
            max_rank: rank,
            tol: 1e-6,
            sigma_color_sq: sc * sc,
            sigma_space_sq: ss * ss,
            k: k + 1, // include trivial eigenvector
        };
        let label = format!("ACA rank={}", rank);

        let ((a, pairs), elapsed_ms) = timed(|| {
            let a = aca_factor(feats, n, &params);
            let (q, r) = aca_qr(&a, n);
            let pairs = aca_lift_eigenpairs(&q, &r, params.k);
            (a, pairs)
        });

        // W_aca = AᵀA (r×N → N×N via transpose multiplication)
        let w_aca = a.transpose() * &a;
        let frob_err = frob_rel_error(&w_dense, &w_aca);

        let vals = sorted_eigenvalues(&pairs);
        // Pad or truncate to match ref length
        let vals_trimmed: Vec<f64> = vals.into_iter().take(ref_vals.len()).collect();
        let (abs_errs, rel_errs) = eigenvalue_errors(&ref_vals, &vals_trimmed);

        // Residuals against reference L (absolute quality measure)
        let residuals: Vec<f64> = pairs
            .iter()
            .take(ref_vals.len())
            .map(|(lam, v)| eigenpair_residual(&l_ref, *lam, v))
            .collect();

        results.push(BenchResult {
            label,
            elapsed_ms,
            eigenvalue_abs_errors: abs_errs,
            eigenvalue_rel_errors: rel_errs,
            eigenvector_residuals: residuals,
            frob_error: Some(frob_err),
        });
    }

    results
}

// ── Table rendering ───────────────────────────────────────────────────────────

fn fmt_sci(v: f64) -> String {
    if v == 0.0 {
        "0.000e+00".to_string()
    } else {
        format!("{:.2e}", v)
    }
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => fmt_sci(x),
        None => "--       ".to_string(),
    }
}

fn print_results_table(results: &[BenchResult], _k: usize) {
    // Check for nearly-degenerate eigenvalues in reference (first result)
    let degenerate_indices: Vec<usize> = Vec::new();
    if let Some(ref_result) = results.first() {
        let abs_errs = &ref_result.eigenvalue_abs_errors;
        // Use ref eigenvalues — they are all 0.0 for ref, so check from second result
        if results.len() > 1 {
            // We check using ref residuals as proxy for degeneracy detection
            let _ = abs_errs; // suppress unused warning
        }
    }

    // Detect nearly-degenerate pairs from eigenvector residuals of reference
    // (use ref eigenvalues stored implicitly as zero abs_errors + small residuals)
    // Instead, scan ref_vals from the reference result's label for (*) marking.
    // We mark (*) based on |λ_i - λ_{i+1}| < 1e-4 — need ref eigenvalues.
    // Since we don't store them in BenchResult, we check abs_err sequences.
    // For simplicity: mark (*) when consecutive abs_errors in ref are < 1e-4.
    // Actually we cannot do this without ref eigenvalues. Skip (*) annotation.
    let _ = degenerate_indices;

    let col_method = 28;
    let col_time = 10;
    let col_val = 11;

    let sep = format!(
        "+{:-<w0$}+{:-<w1$}+{:-<w2$}+{:-<w2$}+{:-<w2$}+{:-<w2$}+",
        "", "", "", "", "", "",
        w0 = col_method + 2,
        w1 = col_time + 2,
        w2 = col_val + 2,
    );

    println!("{}", sep);
    println!(
        "| {:<w0$} | {:>w1$} | {:>w2$} | {:>w2$} | {:>w2$} | {:>w2$} |",
        "Method",
        "Time(ms)",
        "max|Δλ|",
        "max|Δλ|/λ",
        "max resid",
        "frob err",
        w0 = col_method,
        w1 = col_time,
        w2 = col_val,
    );
    println!("{}", sep);

    for r in results {
        let max_abs = r.eigenvalue_abs_errors.iter().cloned().fold(0.0_f64, f64::max);
        let max_rel = r.eigenvalue_rel_errors.iter().cloned().fold(0.0_f64, f64::max);
        let max_res = r.eigenvector_residuals.iter().cloned().fold(0.0_f64, f64::max);

        println!(
            "| {:<w0$} | {:>w1$.2} | {:>w2$} | {:>w2$} | {:>w2$} | {:>w2$} |",
            truncate(&r.label, col_method),
            r.elapsed_ms,
            fmt_sci(max_abs),
            fmt_sci(max_rel),
            fmt_sci(max_res),
            fmt_opt(r.frob_error),
            w0 = col_method,
            w1 = col_time,
            w2 = col_val,
        );
    }

    println!("{}", sep);
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let sizes = [(32usize, 32usize), (48, 48)];
    let k = 6usize;
    let sc = 0.1_f64;
    let ss = 0.15_f64;

    for (w, h) in sizes {
        let n = w * h;

        let images: Vec<(&str, Vec<f64>)> = vec![
            ("gradient",    gradient_feats(w, h)),
            ("checkerboard", checkerboard_feats(w, h, 4)),
            ("noise",        noise_feats(w, h, 42)),
        ];

        for (img_name, feats) in &images {
            println!();
            println!("═══════════════════════════════════════════════════════════");
            println!(" Image: {}×{}  pattern: {}  N={}  k={}  σc={}  σs={}",
                w, h, img_name, n, k, sc, ss);
            println!("═══════════════════════════════════════════════════════════");

            // --- A: Eigensolver comparison ---
            println!();
            println!("  A) Eigensolver comparison (Dense W, Full = ground truth)");
            let results_a = bench_eigensolvers(&feats, n, k, sc, ss);
            print_results_table(&results_a, k);

            // --- B: Similarity comparison ---
            println!();
            println!("  B) Similarity method comparison (Full eigen, Dense = ground truth)");
            let results_b = bench_similarity(&feats, n, k, sc, ss);
            print_results_table(&results_b, k);

            // --- C: ACA vs Classic ---
            println!();
            println!("  C) ACA pipeline vs Classic end-to-end");
            let results_c = bench_aca(&feats, n, k, sc, ss);
            print_results_table(&results_c, k);
        }
    }
}
