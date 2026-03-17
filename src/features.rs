//! Feature extraction: RGB → HSL, build W×H×5 feature matrix.
//!
//! Each pixel becomes a 5-element feature vector [h, s, l, x_norm, y_norm].
//! All values are in [0, 1].  Hue is kept as a raw angle fraction so that the
//! *circular* distance can be computed correctly in the similarity step.

use image::RgbImage;

// ── RGB → HSL ─────────────────────────────────────────────────────────────────

/// Convert an 8-bit RGB triple to HSL, all channels in [0, 1].
///
/// The hue convention follows CSS/Wikipedia:
///   0   = red
///   1/3 = green
///   2/3 = blue
pub fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let r = r as f64 / 255.0;
    let g = g as f64 / 255.0;
    let b = b as f64 / 255.0;

    let cmax = r.max(g).max(b);
    let cmin = r.min(g).min(b);
    let delta = cmax - cmin;

    let l = (cmax + cmin) / 2.0;

    let s = if delta == 0.0 {
        0.0
    } else {
        delta / (1.0 - (2.0 * l - 1.0).abs())
    };

    let h = if delta == 0.0 {
        0.0
    } else if (cmax - r).abs() < 1e-10 {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if (cmax - g).abs() < 1e-10 {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };

    (h, s, l)
}

// ── Feature matrix ────────────────────────────────────────────────────────────

/// Build the W×H×5 feature matrix as a flat `Vec<f64>` of length `W*H*5`.
///
/// Pixel (x, y) maps to slice `[idx*5 .. idx*5+5]` where `idx = y*W + x`.
/// Layout: `[hue, saturation, lightness, x_norm, y_norm]`.
pub fn build_features(img: &RgbImage) -> (Vec<f64>, u32, u32) {
    let (w, h) = img.dimensions();
    let n = (w * h) as usize;
    let mut feats = vec![0f64; n * 5];

    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x, y);
            let (hue, sat, lum) = rgb_to_hsl(px[0], px[1], px[2]);
            let idx = (y * w + x) as usize;
            feats[idx * 5]     = hue;
            feats[idx * 5 + 1] = sat;
            feats[idx * 5 + 2] = lum;
            feats[idx * 5 + 3] = x as f64 / (w - 1).max(1) as f64;
            feats[idx * 5 + 4] = y as f64 / (h - 1).max(1) as f64;
        }
    }
    (feats, w, h)
}

/// Compute the combined (color² / σ_c² + space² / σ_s²) exponent for two pixels.
///
/// Hue is treated as a circular quantity: distance = min(|Δh|, 1 − |Δh|).
/// This helper is shared by both the dense and kNN builders.
pub fn kernel_exponent(
    fi: &[f64],
    fj: &[f64],
    sc2: f64,
    ss2: f64,
) -> f64 {
    let dh = {
        let raw = (fi[0] - fj[0]).abs();
        raw.min(1.0 - raw)
    };
    let d_color2 = dh * dh + (fi[1] - fj[1]).powi(2) + (fi[2] - fj[2]).powi(2);
    let d_space2  = (fi[3] - fj[3]).powi(2) + (fi[4] - fj[4]).powi(2);
    d_color2 / sc2 + d_space2 / ss2
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsl_black() {
        let (_h, s, l) = rgb_to_hsl(0, 0, 0);
        assert!((l - 0.0).abs() < 1e-9);
        assert!((s - 0.0).abs() < 1e-9);
    }

    #[test]
    fn hsl_white() {
        let (_h, s, l) = rgb_to_hsl(255, 255, 255);
        assert!((l - 1.0).abs() < 1e-9);
        assert!((s - 0.0).abs() < 1e-9);
    }

    #[test]
    fn hsl_red() {
        let (h, s, l) = rgb_to_hsl(255, 0, 0);
        assert!((h - 0.0).abs() < 1e-9);
        assert!((s - 1.0).abs() < 1e-9);
        assert!((l - 0.5).abs() < 1e-9);
    }

    #[test]
    fn hsl_cyan() {
        // Cyan = (0,255,255)  → hue = 0.5 (180°)
        let (h, _s, _l) = rgb_to_hsl(0, 255, 255);
        assert!((h - 0.5).abs() < 1e-9, "h={h}");
    }

    #[test]
    fn feature_length() {
        use image::ImageBuffer;
        let img = ImageBuffer::from_fn(4, 3, |x, y| {
            image::Rgb([(x * 60) as u8, (y * 80) as u8, 128u8])
        });
        let (feats, w, h) = build_features(&img);
        assert_eq!(w, 4);
        assert_eq!(h, 3);
        assert_eq!(feats.len(), 4 * 3 * 5);
        // corner pixel (0,0): x_norm=0, y_norm=0
        assert!((feats[3] - 0.0).abs() < 1e-10);
        assert!((feats[4] - 0.0).abs() < 1e-10);
        // corner pixel (3,2): x_norm=1, y_norm=1
        let last = (2 * 4 + 3) * 5;
        assert!((feats[last + 3] - 1.0).abs() < 1e-10);
        assert!((feats[last + 4] - 1.0).abs() < 1e-10);
    }
}
