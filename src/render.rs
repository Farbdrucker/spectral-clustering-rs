//! Rendering: turn eigenvector sign patterns into PNG images.

use image::{ImageBuffer, Rgb, RgbImage};
use nalgebra::DVector;

// ── Sign image ────────────────────────────────────────────────────────────────

/// Render the sign pattern of one eigenvector as a binary image.
///
/// Pixels where the component is ≥ 0 become light grey; pixels with negative
/// components become a saturated blue.  The two regions correspond to the two
/// graph partitions suggested by this eigenvector (Fiedler cut idea).
pub fn render_sign_image(eigvec: &DVector<f64>, width: u32, height: u32) -> RgbImage {
    let mut img = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let px: Rgb<u8> = if eigvec[idx] >= 0.0 {
                Rgb([230u8, 230, 230])  // partition A: light
            } else {
                Rgb([30u8, 65, 175])    // partition B: blue
            };
            img.put_pixel(x, y, px);
        }
    }
    img
}

// ── Summary mosaic ────────────────────────────────────────────────────────────

/// Render all non-trivial eigenvectors side-by-side into one summary PNG.
///
/// The first eigenvector (trivial, λ ≈ 0, constant sign) is skipped.
/// Each panel is labelled with its eigenvector index and eigenvalue.
pub fn render_summary(
    pairs: &[(f64, DVector<f64>)],
    width: u32,
    height: u32,
) -> RgbImage {
    let panels: Vec<&(f64, DVector<f64>)> = pairs.iter().skip(1).collect();
    if panels.is_empty() {
        return ImageBuffer::new(1, 1);
    }

    let pad: u32 = 8;
    let label_h: u32 = 16;
    let cell_w = width + pad;
    let out_w = cell_w * panels.len() as u32 + pad;
    let out_h = height + label_h + 2 * pad;

    let mut out: RgbImage = ImageBuffer::from_pixel(out_w, out_h, Rgb([12u8, 12, 12]));

    for (pos, (val, vec)) in panels.iter().enumerate() {
        let ox = pad + pos as u32 * cell_w;
        let oy = pad + label_h;

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let colour: Rgb<u8> = if vec[idx] >= 0.0 {
                    Rgb([225, 225, 225])
                } else {
                    Rgb([30, 65, 175])
                };
                out.put_pixel(ox + x, oy + y, colour);
            }
        }

        let label = format!("v{}  l={:.4}", pos + 2, val);
        draw_label(&mut out, &label, ox, pad);
    }
    out
}

// ── Pixel-font label renderer ─────────────────────────────────────────────────

/// Burn a short ASCII label into the image using a baked-in 5×7 pixel font.
/// No external font dependency needed.
pub fn draw_label(img: &mut RgbImage, text: &str, x0: u32, y0: u32) {
    let (iw, ih) = img.dimensions();
    let colour = Rgb([210u8, 195, 70]);
    for (ci, ch) in text.chars().enumerate() {
        let glyph = char_glyph(ch);
        for (row, &bits) in glyph.iter().enumerate() {
            for col in 0..5u32 {
                if bits & (1 << (4 - col)) != 0 {
                    let px = x0 + ci as u32 * 6 + col;
                    let py = y0 + row as u32;
                    if px < iw && py < ih {
                        img.put_pixel(px, py, colour);
                    }
                }
            }
        }
    }
}

/// 5×7 bitmapped glyphs.  Each byte is one row, 5 bits wide (MSB = left).
fn char_glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0x0E,0x11,0x13,0x15,0x19,0x11,0x0E],
        '1' => [0x04,0x0C,0x04,0x04,0x04,0x04,0x0E],
        '2' => [0x0E,0x11,0x01,0x06,0x08,0x10,0x1F],
        '3' => [0x1F,0x02,0x04,0x02,0x01,0x11,0x0E],
        '4' => [0x02,0x06,0x0A,0x12,0x1F,0x02,0x02],
        '5' => [0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E],
        '6' => [0x06,0x08,0x10,0x1E,0x11,0x11,0x0E],
        '7' => [0x1F,0x01,0x02,0x04,0x08,0x08,0x08],
        '8' => [0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E],
        '9' => [0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C],
        'v' => [0x00,0x11,0x11,0x0A,0x0A,0x04,0x00],
        'l' => [0x06,0x02,0x02,0x02,0x02,0x02,0x07],
        '=' => [0x00,0x1F,0x00,0x1F,0x00,0x00,0x00],
        '.' => [0x00,0x00,0x00,0x00,0x00,0x0C,0x0C],
        '-' => [0x00,0x00,0x00,0x1F,0x00,0x00,0x00],
        ' ' => [0x00,0x00,0x00,0x00,0x00,0x00,0x00],
        _   => [0x00,0x0E,0x0A,0x0E,0x08,0x08,0x00],
    }
}

// ── Continuous-value image (conv-split output) ────────────────────────────────

/// Render a continuous field u ∈ [-1, +1] as a colour image.
///
/// Mapping:
///   u =  1  → warm white  (foreground, partition A)
///   u =  0  → mid grey
///   u = -1  → deep blue   (background, partition B)
///
/// Values outside [-1, +1] are clamped.
pub fn render_continuous(u: &DVector<f64>, width: u32, height: u32) -> RgbImage {
    let mut img = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let v = u[idx].clamp(-1.0, 1.0);
            // Interpolate: -1 → blue(30,65,175), 0 → grey(128,128,128), +1 → white(235,230,220)
            let px: Rgb<u8> = if v >= 0.0 {
                // 0 → grey, 1 → warm white
                let t = v as f32;
                Rgb([
                    lerp(128, 235, t),
                    lerp(128, 230, t),
                    lerp(128, 220, t),
                ])
            } else {
                // -1 → blue, 0 → grey
                let t = (-v) as f32;
                Rgb([
                    lerp(128, 30, t),
                    lerp(128, 65, t),
                    lerp(128, 175, t),
                ])
            };
            img.put_pixel(x, y, px);
        }
    }
    img
}

/// Render u as a hard binary image by thresholding at 0.
pub fn render_binary(u: &DVector<f64>, width: u32, height: u32) -> RgbImage {
    render_sign_image(u, width, height) // reuse sign renderer
}

/// Render the label seed image u₀ so the user can verify input.
///
/// +1 → bright green,  -1 → bright red,  0 → dark grey.
pub fn render_u0(u0: &[f64], width: u32, height: u32) -> RgbImage {
    let mut img = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let v = u0[idx];
            let px: Rgb<u8> = if v > 0.5 {
                Rgb([60u8, 220, 80])    // green → foreground seed
            } else if v < -0.5 {
                Rgb([220u8, 50, 50])    // red   → background seed
            } else {
                Rgb([40u8, 40, 40])     // dark  → unlabelled
            };
            img.put_pixel(x, y, px);
        }
    }
    img
}

#[inline]
fn lerp(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t.clamp(0.0, 1.0)) as u8
}
