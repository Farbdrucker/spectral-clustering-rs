//! GUI parameter tuner for spectral-clustering-rs.
//!
//! Opens a native window with sliders/dropdowns for the six key parameters.
//! Whenever a parameter changes, a background thread recomputes the segmentation
//! masks and displays them in the central panel.
//!
//! Usage:
//!   cargo run --release --bin gui

use std::path::PathBuf;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

use eframe::egui::{self, Color32, TextureOptions};
use image::{imageops::FilterType, RgbImage};

use spectral_segment::{
    aca::{aca_factor, aca_lift_eigenpairs, aca_qr, AcaParams},
    eigen::{compute_eigenpairs, EigenMethod},
    features::build_features,
    laplacian::normalised_laplacian,
    render::render_sign_image,
    similarity::{build_similarity, SimMethod},
};

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Spectral Clustering Tuner")
            .with_inner_size([1200.0, 750.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Spectral Clustering Tuner",
        options,
        Box::new(|cc| Ok(Box::new(GuiApp::new(cc)))),
    )
}

// ── Local parameter enums (PartialEq — not clap) ──────────────────────────────

#[derive(Clone, PartialEq)]
enum PipelineChoice {
    Classic,
    Aca,
}

#[derive(Clone, PartialEq)]
enum SimMethodChoice {
    Dense,
    Knn,
    Nystrom,
}

#[derive(Clone, PartialEq)]
enum EigenMethodChoice {
    Full,
    PowerIter,
    Lanczos,
}

// ── Tunable parameters ────────────────────────────────────────────────────────

#[derive(Clone, PartialEq)]
struct Params {
    sigma_color:  f64,
    sigma_space:  f64,
    max_side:     u32,
    pipeline:     PipelineChoice,
    sim_method:   SimMethodChoice,
    eigen_method: EigenMethodChoice,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            sigma_color:  0.1,
            sigma_space:  0.15,
            max_side:     48,
            pipeline:     PipelineChoice::Classic,
            sim_method:   SimMethodChoice::Dense,
            eigen_method: EigenMethodChoice::Full,
        }
    }
}

// ── Channel messages ──────────────────────────────────────────────────────────

struct ComputeRequest {
    image:  RgbImage,
    params: Params,
}

struct ComputeResult {
    images:     Vec<RgbImage>,
    elapsed_ms: f64,
    error:      Option<String>,
}

// ── App state ─────────────────────────────────────────────────────────────────

struct GuiApp {
    image_path:  Option<PathBuf>,
    raw_image:   Option<RgbImage>,
    params:      Params,
    last_params: Option<Params>,
    req_tx:      SyncSender<ComputeRequest>,
    res_rx:      Receiver<ComputeResult>,
    textures:    Vec<egui::TextureHandle>,
    status:      String,
    computing:   bool,
}

impl GuiApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (req_tx, req_rx) = sync_channel::<ComputeRequest>(1);
        let (res_tx, res_rx) = sync_channel::<ComputeResult>(4);

        let ctx = cc.egui_ctx.clone();
        std::thread::spawn(move || {
            loop {
                // Block until a request arrives.
                let mut req = match req_rx.recv() {
                    Ok(r)  => r,
                    Err(_) => break, // sender dropped → exit thread
                };
                // Drain stale requests — always process the newest.
                while let Ok(newer) = req_rx.try_recv() {
                    req = newer;
                }

                let t0 = std::time::Instant::now();
                let result = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| run_pipeline(&req.image, &req.params)),
                );
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let cr = match result {
                    Ok(Ok(images))  => ComputeResult { images,     elapsed_ms, error: None },
                    Ok(Err(e))      => ComputeResult { images: vec![], elapsed_ms, error: Some(e) },
                    Err(_)          => ComputeResult { images: vec![], elapsed_ms,
                                           error: Some("Pipeline panicked".into()) },
                };
                let _ = res_tx.send(cr);
                ctx.request_repaint();
            }
        });

        Self {
            image_path:  None,
            raw_image:   None,
            params:      Params::default(),
            last_params: None,
            req_tx,
            res_rx,
            textures:    vec![],
            status:      "Open an image to begin.".into(),
            computing:   false,
        }
    }
}

// ── UI ────────────────────────────────────────────────────────────────────────

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── Poll for completed results ─────────────────────────────────────────
        if let Ok(result) = self.res_rx.try_recv() {
            self.computing = false;
            if let Some(err) = result.error {
                self.status = format!("Error: {}", err);
                self.textures.clear();
            } else {
                self.status = format!("Done in {:.0} ms", result.elapsed_ms);
                self.textures = result.images.iter().enumerate()
                    .map(|(i, img)| {
                        let cimg = rgb_to_egui(img);
                        ctx.load_texture(
                            format!("eigvec_{}", i),
                            cimg,
                            TextureOptions::NEAREST,
                        )
                    })
                    .collect();
            }
        }

        // ── Top panel — toolbar ───────────────────────────────────────────────
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open Image…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tiff", "tif"])
                        .pick_file()
                    {
                        match image::open(&path) {
                            Ok(dyn_img) => {
                                self.raw_image   = Some(dyn_img.into_rgb8());
                                self.image_path  = Some(path);
                                self.last_params = None; // force immediate recompute
                                self.textures.clear();
                                self.status = "Image loaded.".into();
                            }
                            Err(e) => {
                                self.status = format!("Failed to open image: {}", e);
                            }
                        }
                    }
                }

                if let Some(p) = &self.image_path {
                    ui.label(p.file_name().and_then(|n| n.to_str()).unwrap_or("?"));
                }

                ui.separator();
                ui.label(&self.status);
                if self.computing {
                    ui.spinner();
                }
            });
        });

        // ── Left panel — parameter controls ──────────────────────────────────
        egui::SidePanel::left("controls").min_width(230.0).show(ctx, |ui| {
            ui.heading("Parameters");
            ui.add_space(8.0);

            // σ color — logarithmic slider
            ui.label("σ color");
            ui.add(
                egui::Slider::new(&mut self.params.sigma_color, 0.001..=0.5)
                    .logarithmic(true)
                    .fixed_decimals(4),
            );
            ui.add_space(4.0);

            // σ space — logarithmic slider
            ui.label("σ space");
            ui.add(
                egui::Slider::new(&mut self.params.sigma_space, 0.001..=1.0)
                    .logarithmic(true)
                    .fixed_decimals(4),
            );
            ui.add_space(4.0);

            // max-side
            ui.label("Max side (px)");
            ui.add(egui::DragValue::new(&mut self.params.max_side).range(16..=256));
            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            // Pipeline
            ui.label("Pipeline");
            egui::ComboBox::from_id_source("pipeline")
                .selected_text(match self.params.pipeline {
                    PipelineChoice::Classic => "Classic",
                    PipelineChoice::Aca     => "ACA",
                })
                .show_ui(ui, |ui: &mut egui::Ui| {
                    ui.selectable_value(&mut self.params.pipeline, PipelineChoice::Classic, "Classic");
                    ui.selectable_value(&mut self.params.pipeline, PipelineChoice::Aca,     "ACA");
                });

            if self.params.pipeline == PipelineChoice::Classic {
                ui.add_space(4.0);

                // Affinity method
                ui.label("Affinity");
                egui::ComboBox::from_id_source("sim_method")
                    .selected_text(match self.params.sim_method {
                        SimMethodChoice::Dense   => "Dense",
                        SimMethodChoice::Knn     => "Knn",
                        SimMethodChoice::Nystrom => "Nystrom",
                    })
                    .show_ui(ui, |ui: &mut egui::Ui| {
                        ui.selectable_value(&mut self.params.sim_method, SimMethodChoice::Dense,   "Dense");
                        ui.selectable_value(&mut self.params.sim_method, SimMethodChoice::Knn,     "Knn");
                        ui.selectable_value(&mut self.params.sim_method, SimMethodChoice::Nystrom, "Nystrom");
                    });

                ui.add_space(4.0);

                // Eigensolver
                ui.label("Eigensolver");
                egui::ComboBox::from_id_source("eigen_method")
                    .selected_text(match self.params.eigen_method {
                        EigenMethodChoice::Full      => "Full",
                        EigenMethodChoice::PowerIter => "PowerIter",
                        EigenMethodChoice::Lanczos   => "Lanczos",
                    })
                    .show_ui(ui, |ui: &mut egui::Ui| {
                        ui.selectable_value(&mut self.params.eigen_method, EigenMethodChoice::Full,      "Full");
                        ui.selectable_value(&mut self.params.eigen_method, EigenMethodChoice::PowerIter, "PowerIter");
                        ui.selectable_value(&mut self.params.eigen_method, EigenMethodChoice::Lanczos,   "Lanczos");
                    });
            } else {
                ui.add_space(4.0);
                ui.label("Rank = ceil(√N)  (auto)");
            }
        });

        // ── Central panel — eigenvector sign patterns ─────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.textures.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open an image to begin.");
                });
            } else {
                let avail  = ui.available_size();
                let cols   = 3usize;
                let rows   = (self.textures.len() + cols - 1) / cols;
                let cell_w = (avail.x / cols as f32 - 4.0).max(1.0);
                let cell_h = (avail.y / rows as f32 - 4.0).max(1.0);

                egui::Grid::new("eigvec_grid")
                    .num_columns(cols)
                    .spacing([4.0, 4.0])
                    .show(ui, |ui| {
                        for (i, tex) in self.textures.iter().enumerate() {
                            let sz    = tex.size_vec2();
                            let scale = (cell_w / sz.x).min(cell_h / sz.y).min(8.0);
                            ui.image((tex.id(), sz * scale));
                            if (i + 1) % cols == 0 {
                                ui.end_row();
                            }
                        }
                    });
            }
        });

        // ── Hot-reload trigger ────────────────────────────────────────────────
        let should_compute = self.raw_image.is_some()
            && !self.computing
            && (self.last_params.is_none()
                || self.last_params.as_ref().map(|p| p != &self.params).unwrap_or(true));

        if should_compute {
            self.last_params = Some(self.params.clone());
            let req = ComputeRequest {
                image:  self.raw_image.clone().unwrap(),
                params: self.params.clone(),
            };
            let _ = self.req_tx.try_send(req);
            self.computing = true;
            self.status = "Computing…".into();
        }
    }
}

// ── Pipeline runner (called inside worker thread) ─────────────────────────────

fn run_pipeline(img: &RgbImage, params: &Params) -> Result<Vec<RgbImage>, String> {
    let scaled      = downscale(img, params.max_side);
    let (w, h)      = scaled.dimensions();
    let n           = (w * h) as usize;
    if n == 0 {
        return Err("Empty image after downscale".into());
    }

    let (feats, _, _) = build_features(&scaled);
    let k_total       = 7usize; // eigenvectors to compute: 1 trivial + 6 non-trivial

    let pairs = match params.pipeline {
        // ── ACA pipeline ──────────────────────────────────────────────────────
        PipelineChoice::Aca => {
            let max_rank = ((n as f64).sqrt().ceil() as usize).max(k_total + 2);
            let aca_p = AcaParams {
                max_rank,
                tol: 1e-6,
                sigma_color_sq: params.sigma_color * params.sigma_color,
                sigma_space_sq: params.sigma_space * params.sigma_space,
                k: k_total,
            };
            let a    = aca_factor(&feats, n, &aca_p);
            let rank = a.nrows();
            if rank == 0 {
                return Err("ACA returned rank-0 factor — degenerate image?".into());
            }
            let (q, r) = aca_qr(&a, n);
            aca_lift_eigenpairs(&q, &r, k_total)
        }

        // ── Classic pipeline ──────────────────────────────────────────────────
        PipelineChoice::Classic => {
            let nystrom_m = ((n as f64).sqrt().ceil() as usize).max(8);
            let sim_method = match params.sim_method {
                SimMethodChoice::Dense   => SimMethod::Dense,
                SimMethodChoice::Knn     => SimMethod::Knn,
                SimMethodChoice::Nystrom => SimMethod::Nystrom,
            };
            let sim = build_similarity(
                &feats, n,
                params.sigma_color, params.sigma_space,
                &sim_method, 20, nystrom_m,
            );
            let lap = normalised_laplacian(&sim);
            let eigen_method = match params.eigen_method {
                EigenMethodChoice::Full      => EigenMethod::Full,
                EigenMethodChoice::PowerIter => EigenMethod::PowerIter,
                EigenMethodChoice::Lanczos   => EigenMethod::Lanczos,
            };
            let lanczos_q = (3 * k_total + 20).max(30).min(n);
            compute_eigenpairs(&lap, k_total, &eigen_method, 500, 1e-7, lanczos_q)
        }
    };

    // Skip trivial eigenvector 0 (λ ≈ 0, constant sign) and render the rest.
    let images = pairs.iter().skip(1)
        .map(|(_, vec)| render_sign_image(vec, w, h))
        .collect();
    Ok(images)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn downscale(img: &RgbImage, max_side: u32) -> RgbImage {
    let (w, h)  = img.dimensions();
    let longest = w.max(h);
    if longest <= max_side {
        return img.clone();
    }
    let s  = max_side as f64 / longest as f64;
    let nw = ((w as f64 * s).round() as u32).max(1);
    let nh = ((h as f64 * s).round() as u32).max(1);
    image::imageops::resize(img, nw, nh, FilterType::Lanczos3)
}

fn rgb_to_egui(img: &RgbImage) -> egui::ColorImage {
    egui::ColorImage {
        size:   [img.width() as usize, img.height() as usize],
        pixels: img.pixels().map(|p| Color32::from_rgb(p[0], p[1], p[2])).collect(),
    }
}
