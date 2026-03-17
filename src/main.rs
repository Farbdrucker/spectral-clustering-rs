//! # spectral_segment
//!
//! Spectral image segmentation via the normalised graph Laplacian,
//! with optional semi-supervised refinement via Allen-Cahn convexity splitting.
//!
//! ## Timing
//!
//! Every significant computational step is timed independently.
//! A formatted table is printed to stderr at the end of the run, and
//! `timing.json` is written to the output directory for cross-run analysis.
//!
//! **ACA sub-stages timed separately:**
//!   - `aca::factor`      — pivoted Cholesky, O(N·r) kernel evaluations
//!   - `aca::qr`          — degree normalisation + Householder QR, O(N·r²)
//!   - `aca::eigen_r`     — symmetric eigen of r×r matrix, O(r³)
//!
//! **Classic pipeline sub-stages:**
//!   - `similarity::{dense|knn|nystrom}`
//!   - `laplacian::normalised`
//!   - `eigensolver::{full|power_iter|lanczos}`
//!
//! **Conv-split:**
//!   - `conv_split::load_u0`
//!   - `conv_split::solver`
//!   - `render::conv_split`

mod aca;
mod conv_split;
mod eigen;
mod features;
mod laplacian;
mod render;
mod similarity;
mod timing;

use clap::Parser;
use nalgebra::{DMatrix, DVector};
use std::path::PathBuf;

use aca::{aca_factor, aca_lift_eigenpairs, aca_qr, AcaParams};
use conv_split::{conv_splitting, load_u0_from_png, ConvSplitParams};
use eigen::{compute_eigenpairs, EigenMethod};
use features::build_features;
use laplacian::normalised_laplacian;
use render::{
    draw_label, render_binary, render_continuous, render_sign_image, render_summary, render_u0,
};
use similarity::{build_similarity, SimMethod};
use timing::Timer;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, clap::ValueEnum, PartialEq)]
enum Pipeline { Classic, Aca }

#[derive(Parser, Debug)]
#[command(
    name = "spectral_segment",
    about = "Spectral segmentation + semi-supervised refinement with precise stage timing",
    long_about = "\
Spectral image segmentation via the normalised graph Laplacian, with optional
semi-supervised refinement via Allen-Cahn convexity splitting (Stoll/Buenger 2016).

TIMING
  Every computational sub-task is timed independently.  At the end of each run:
  - A table is printed to stderr with per-stage durations and cumulative %.
  - timing.json is written to the output directory.

  Load in Python for cross-run comparison:
    import json, pandas as pd
    df = pd.DataFrame(json.load(open('out/timing.json'))['entries'])
    print(df[['name','params','ms']].to_string())

SPECTRUM PIPELINES (--pipeline)
  classic  features → W (dense|knn|nystrom) → L → eigensolver (full|power|lanczos)
           Timed stages: similarity::<method>, laplacian::normalised,
                         eigensolver::<method>
  aca      features → ACA factor → QR → r×r eigen
           Timed stages: aca::factor, aca::qr, aca::eigen_r
           Never forms W or L.  O(N·r²) time and O(N·r) memory.

SEMI-SUPERVISED REFINEMENT (--u0)
  Supply a greyscale PNG: white(>=192)->+1  black(<=63)->-1  grey->0(free)
  Timed stages: conv_split::load_u0, conv_split::solver, render::conv_split

RECOMMENDED SETTINGS
  max-side <= 32  : classic dense     + full
  max-side <= 64  : classic knn       + lanczos  --lanczos-q 40
  max-side <= 128 : aca               (rank = ceil(sqrt(N)))
  max-side > 128  : aca               --aca-max-rank 80
"
)]
struct Args {
    // ── I/O ──────────────────────────────────────────────────────────────────
    /// Input image (any format supported by the image crate)
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory (timing.json written here)
    #[arg(short, long, default_value = "out")]
    output: PathBuf,

    // ── Shared ───────────────────────────────────────────────────────────────
    #[arg(long, value_enum, default_value_t = Pipeline::Classic)]
    pipeline: Pipeline,

    /// Non-trivial eigenvectors to render (trivial lambda≈0 excluded from output)
    #[arg(short = 'n', long, default_value_t = 6)]
    num_eigenvectors: usize,

    /// Downscale: longest edge <= this many pixels (controls N = W*H)
    #[arg(long, default_value_t = 48)]
    max_side: u32,

    /// sigma for HSL colour distance in kernel
    #[arg(long, default_value_t = 0.1)]
    sigma_color: f64,

    /// sigma for XY spatial distance in kernel
    #[arg(long, default_value_t = 0.15)]
    sigma_space: f64,

    // ── ACA ──────────────────────────────────────────────────────────────────
    /// [aca] Max ACA rank.  0 => auto ceil(sqrt(N))
    #[arg(long, default_value_t = 0)]
    aca_max_rank: usize,

    /// [aca] Diagonal-remainder stop tolerance
    #[arg(long, default_value_t = 1e-6)]
    aca_tol: f64,

    // ── Classic similarity ────────────────────────────────────────────────────
    #[arg(long, value_enum, default_value_t = SimMethod::Dense)]
    sim_method: SimMethod,

    /// [knn] Neighbours per pixel
    #[arg(long, default_value_t = 20)]
    knn_k: usize,

    /// [nystrom] Landmark count.  0 => auto ceil(sqrt(N))
    #[arg(long, default_value_t = 0)]
    nystrom_m: usize,

    // ── Classic eigensolver ───────────────────────────────────────────────────
    #[arg(long, value_enum, default_value_t = EigenMethod::Full)]
    eigen_method: EigenMethod,

    #[arg(long, default_value_t = 500)]
    power_max_iter: usize,

    #[arg(long, default_value_t = 1e-7)]
    power_tol: f64,

    /// [lanczos] Krylov subspace dimension.  0 => auto min(N, max(3k+20, 30))
    #[arg(long, default_value_t = 0)]
    lanczos_q: usize,

    // ── Conv-split ────────────────────────────────────────────────────────────
    /// Label image path (greyscale PNG: white=+1, black=-1, grey=0)
    #[arg(long)]
    u0: Option<PathBuf>,

    /// [conv-split] Fidelity weight omega_0
    #[arg(long, default_value_t = 1.0)]
    cs_omega0: f64,

    /// [conv-split] Interface width epsilon.  0 => auto 1/sqrt(N)
    #[arg(long, default_value_t = 0.0)]
    cs_eps: f64,

    /// [conv-split] Time step dt
    #[arg(long, default_value_t = 0.01)]
    cs_dt: f64,

    /// [conv-split] Stabilisation constant c.  0 => auto 3/eps + omega_0
    #[arg(long, default_value_t = 0.0)]
    cs_c: f64,

    #[arg(long, default_value_t = 500)]
    cs_max_iter: usize,

    #[arg(long, default_value_t = 1e-8)]
    cs_tol: f64,

    /// [conv-split] Print residual every iteration
    #[arg(long, default_value_t = false)]
    cs_verbose: bool,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn downscale(img: &image::RgbImage, max_side: u32) -> image::RgbImage {
    let (w, h) = img.dimensions();
    let longest = w.max(h);
    if longest <= max_side { return img.clone(); }
    let scale = max_side as f64 / longest as f64;
    let nw = ((w as f64 * scale).round() as u32).max(1);
    let nh = ((h as f64 * scale).round() as u32).max(1);
    let mut out = image::ImageBuffer::new(nw, nh);
    for ny in 0..nh {
        for nx in 0..nw {
            let ox = (nx as f64 / nw as f64 * w as f64) as u32;
            let oy = (ny as f64 / nh as f64 * h as f64) as u32;
            out.put_pixel(nx, ny, *img.get_pixel(ox.min(w-1), oy.min(h-1)));
        }
    }
    out
}

fn sim_tag(m: &SimMethod) -> &'static str {
    match m { SimMethod::Dense=>"dense", SimMethod::Knn=>"knn", SimMethod::Nystrom=>"nystrom" }
}
fn eigen_tag(m: &EigenMethod) -> &'static str {
    match m { EigenMethod::Full=>"full", EigenMethod::PowerIter=>"power_iter", EigenMethod::Lanczos=>"lanczos" }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut t = Timer::new();

    // ── 1. Load & downscale ───────────────────────────────────────────────────
    eprintln!("── I/O ────────────────────────────────────────────────────────");

    let raw = t.measure(
        "io::load_image",
        format!("file={}", args.input.file_name()
            .and_then(|n| n.to_str()).unwrap_or("?")),
        || image::open(&args.input)
               .map_err(|e| anyhow::anyhow!("{e}"))
               .unwrap()
               .into_rgb8(),
    );

    let img = t.measure(
        "io::downscale",
        format!("max_side={}", args.max_side),
        || downscale(&raw, args.max_side),
    );

    let (w, h) = img.dimensions();
    let n = (w * h) as usize;
    eprintln!("   {}×{} = {} pixels  pipeline={:?}", w, h, n, args.pipeline);

    // ── 2. Features ───────────────────────────────────────────────────────────
    eprintln!("── Features ───────────────────────────────────────────────────");

    let (feats, _, _) = t.measure(
        "features::hsl_xy",
        format!("N={n}"),
        || build_features(&img),
    );

    // ── 3+4. Eigenpairs ───────────────────────────────────────────────────────
    let k_total = args.num_eigenvectors + 1; // +1 for trivial λ≈0

    let pairs: Vec<(f64, DVector<f64>)> = match args.pipeline {

        // ── ACA — three timed sub-stages ─────────────────────────────────────
        Pipeline::Aca => {
            let max_rank = if args.aca_max_rank == 0 {
                ((n as f64).sqrt().ceil() as usize).max(k_total + 2)
            } else { args.aca_max_rank };

            let aca_p = AcaParams {
                max_rank,
                tol: args.aca_tol,
                sigma_color_sq: args.sigma_color * args.sigma_color,
                sigma_space_sq: args.sigma_space * args.sigma_space,
                k: k_total,
            };
            let p_str = format!(
                "N={n} max_rank={max_rank} tol={:.0e} sc={:.3} ss={:.3}",
                args.aca_tol, args.sigma_color, args.sigma_space
            );

            eprintln!("── ACA::factor ────────────────────────────────────────────────");
            let a = t.measure("aca::factor", &p_str, || aca_factor(&feats, n, &aca_p));
            let rank = a.nrows();
            eprintln!("   rank = {rank}");

            if rank == 0 {
                eprintln!("   WARNING: ACA returned rank-0 factor — degenerate input?");
                let v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
                vec![(0.0, v)]
            } else {
                eprintln!("── ACA::qr  (degree norm + Householder QR) ────────────────────");
                let (q, r) = t.measure(
                    "aca::qr",
                    format!("N={n} r={rank}"),
                    || aca_qr(&a, n),
                );

                eprintln!("── ACA::eigen_r  ({}×{} eigenproblem) ─────────────────────────", rank, rank);
                t.measure(
                    "aca::eigen_r",
                    format!("r={rank} k={k_total}"),
                    || aca_lift_eigenpairs(&q, &r, k_total),
                )
            }
        }

        // ── Classic — three independently timed stages ───────────────────────
        Pipeline::Classic => {
            let nystrom_m = if args.nystrom_m == 0 {
                ((n as f64).sqrt().ceil() as usize).max(8)
            } else { args.nystrom_m };

            // Similarity
            eprintln!("── Similarity  method={} ──────────────────────────────────────",
                      sim_tag(&args.sim_method));
            let sim_p = match &args.sim_method {
                SimMethod::Dense   => format!("N={n} sc={:.3} ss={:.3}",
                                              args.sigma_color, args.sigma_space),
                SimMethod::Knn     => format!("N={n} k={} sc={:.3} ss={:.3}",
                                              args.knn_k, args.sigma_color, args.sigma_space),
                SimMethod::Nystrom => format!("N={n} m={nystrom_m} sc={:.3} ss={:.3}",
                                              args.sigma_color, args.sigma_space),
            };
            let sim = t.measure(
                format!("similarity::{}", sim_tag(&args.sim_method)),
                &sim_p,
                || build_similarity(&feats, n, args.sigma_color, args.sigma_space,
                                    &args.sim_method, args.knn_k, nystrom_m),
            );

            // Laplacian
            eprintln!("── Laplacian ───────────────────────────────────────────────────");
            let lap = t.measure(
                "laplacian::normalised",
                format!("N={n}"),
                || normalised_laplacian(&sim),
            );

            // Eigensolver
            let lanczos_q = if args.lanczos_q == 0 {
                (3 * k_total + 20).max(30).min(n)
            } else { args.lanczos_q };

            eprintln!("── Eigensolver  method={} ─────────────────────────────────────",
                      eigen_tag(&args.eigen_method));
            let eigen_p = match &args.eigen_method {
                EigenMethod::Full      => format!("N={n} k={k_total}"),
                EigenMethod::PowerIter => format!("N={n} k={k_total} max_iter={} tol={:.0e}",
                                                  args.power_max_iter, args.power_tol),
                EigenMethod::Lanczos   => format!("N={n} k={k_total} q={lanczos_q}"),
            };
            t.measure(
                format!("eigensolver::{}", eigen_tag(&args.eigen_method)),
                &eigen_p,
                || compute_eigenpairs(&lap, k_total, &args.eigen_method,
                                      args.power_max_iter, args.power_tol, lanczos_q),
            )
        }
    };

    for (i, (val, _)) in pairs.iter().enumerate() {
        eprintln!("   eigenpair {:>2}: λ = {:.8}", i, val);
    }

    // ── 5. Render spectrum ────────────────────────────────────────────────────
    eprintln!("── Render ─────────────────────────────────────────────────────");
    std::fs::create_dir_all(&args.output)?;

    t.measure(
        "render::spectrum",
        format!("{} eigvec images + summary", pairs.len()),
        || -> anyhow::Result<()> {
            for (i, (val, vec)) in pairs.iter().enumerate() {
                render_sign_image(vec, w, h)
                    .save(args.output.join(format!("eigvec_{:02}.png", i)))?;
            }
            render_summary(&pairs, w, h).save(args.output.join("summary.png"))?;
            eprintln!("   {} images written", pairs.len() + 1);
            Ok(())
        }
    )?;

    // ── 6. Optional conv-split ────────────────────────────────────────────────
    if let Some(u0_path) = &args.u0 {
        run_conv_split(&args, u0_path, &pairs, w, h, n, &mut t)?;
    }

    // ── Timing report ─────────────────────────────────────────────────────────
    t.print_table();
    t.write_json(&args.output)?;

    eprintln!("Done.");
    Ok(())
}

// ── Conv-split stage ──────────────────────────────────────────────────────────

fn run_conv_split(
    args: &Args,
    u0_path: &PathBuf,
    pairs: &[(f64, DVector<f64>)],
    w: u32, h: u32, n: usize,
    t: &mut Timer,
) -> anyhow::Result<()> {
    eprintln!("── Conv-split ──────────────────────────────────────────────────");

    let u0 = t.measure(
        "conv_split::load_u0",
        format!("file={}", u0_path.file_name()
            .and_then(|n| n.to_str()).unwrap_or("?")),
        || load_u0_from_png(u0_path, w, h).expect("load_u0 failed"),
    );

    let n_pos  = u0.iter().filter(|&&v| v >  0.5).count();
    let n_neg  = u0.iter().filter(|&&v| v < -0.5).count();
    let n_free = n - n_pos - n_neg;
    eprintln!("   +1={n_pos}  -1={n_neg}  free={n_free}");

    if n_pos == 0 && n_neg == 0 {
        eprintln!("   WARNING: no labelled pixels — skipping conv-split.");
        return Ok(());
    }

    t.measure("render::u0_seeds", format!("N={n}"), || {
        render_u0(&u0, w, h)
            .save(args.output.join("u0_seeds.png"))
            .expect("save failed");
    });

    // Assemble Φ and λ
    let k = pairs.len();
    let mut phi = DMatrix::<f64>::zeros(n, k);
    let mut lambdas = Vec::with_capacity(k);
    for (col, (lambda, vec)) in pairs.iter().enumerate() {
        lambdas.push(*lambda);
        for row in 0..n { phi[(row, col)] = vec[row]; }
    }

    let cs_params = ConvSplitParams {
        omega0:   args.cs_omega0,
        eps:      if args.cs_eps == 0.0 { None } else { Some(args.cs_eps) },
        dt:       args.cs_dt,
        c:        if args.cs_c   == 0.0 { None } else { Some(args.cs_c) },
        max_iter: args.cs_max_iter,
        tol:      args.cs_tol,
        verbose:  args.cs_verbose,
    };
    let eps_v = cs_params.eps.unwrap_or(1.0 / (n as f64).sqrt());
    let c_v   = cs_params.c  .unwrap_or(3.0 / eps_v + cs_params.omega0);
    let cs_p  = format!(
        "N={n} k={k} omega0={} eps={:.4} c={:.4} dt={} maxIter={}",
        cs_params.omega0, eps_v, c_v, cs_params.dt, cs_params.max_iter
    );

    let result = t.measure("conv_split::solver", &cs_p, || {
        conv_splitting(&u0, &lambdas, &phi, &cs_params)
    });
    eprintln!("   iters={}  residual={:.3e}", result.iters + 1, result.final_residual);

    t.measure("render::conv_split", format!("N={n}"), || -> anyhow::Result<()> {
        render_continuous(&result.u, w, h)
            .save(args.output.join("conv_split_continuous.png"))?;
        render_binary(&result.u, w, h)
            .save(args.output.join("conv_split_binary.png"))?;
        save_conv_split_mosaic(
            args,
            &pairs.iter().skip(1).collect::<Vec<_>>(),
            &render_continuous(&result.u, w, h),
            &render_binary(&result.u, w, h),
            &u0, w, h,
        )?;
        eprintln!("   conv_split_{{continuous,binary,mosaic}}.png");
        Ok(())
    })?;

    Ok(())
}

// ── Conv-split mosaic ─────────────────────────────────────────────────────────

fn save_conv_split_mosaic(
    args: &Args,
    non_trivial: &[&(f64, DVector<f64>)],
    cont: &image::RgbImage,
    bin: &image::RgbImage,
    u0: &[f64],
    w: u32, h: u32,
) -> anyhow::Result<()> {
    let pad: u32 = 8;
    let lh:  u32 = 16;
    let n_panels = 1 + non_trivial.len() + 2;
    let cell_w   = w + pad;
    let out_w    = cell_w * n_panels as u32 + pad;
    let out_h    = h + lh + 2 * pad;
    let mut out: image::RgbImage =
        image::ImageBuffer::from_pixel(out_w, out_h, image::Rgb([12u8, 12, 12]));

    let oy  = pad + lh;
    let mut col = 0u32;

    // seeds
    blit(&mut out, &render_u0(u0, w, h), pad + col * cell_w, oy);
    draw_label(&mut out, "seeds", pad + col * cell_w, pad);
    col += 1;

    // eigvec signs
    for (pos, (val, vec)) in non_trivial.iter().enumerate() {
        let ox = pad + col * cell_w;
        blit(&mut out, &render_sign_image(vec, w, h), ox, oy);
        draw_label(&mut out, &format!("v{}  {:.3}", pos+2, val), ox, pad);
        col += 1;
    }

    // continuous u
    blit(&mut out, cont, pad + col * cell_w, oy);
    draw_label(&mut out, "u cont", pad + col * cell_w, pad);
    col += 1;

    // binary u
    blit(&mut out, bin, pad + col * cell_w, oy);
    draw_label(&mut out, "u bin", pad + col * cell_w, pad);

    out.save(args.output.join("conv_split_mosaic.png"))?;
    Ok(())
}

fn blit(dst: &mut image::RgbImage, src: &image::RgbImage, ox: u32, oy: u32) {
    let (sw, sh) = src.dimensions();
    let (dw, dh) = dst.dimensions();
    for y in 0..sh {
        for x in 0..sw {
            let (dx, dy) = (ox + x, oy + y);
            if dx < dw && dy < dh {
                dst.put_pixel(dx, dy, *src.get_pixel(x, y));
            }
        }
    }
}
