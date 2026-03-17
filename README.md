# spectral-clustering-rs

![CI](https://github.com/Farbdrucker/spectral-clustering-rs/actions/workflows/ci.yml/badge.svg)
![Rust](https://img.shields.io/badge/rust-2021-orange?logo=rust)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)
![nalgebra](https://img.shields.io/badge/nalgebra-0.32-lightgrey)

Spectral image segmentation in Rust. Represents an image as a weighted pixel-similarity graph, computes eigenvectors of the normalised graph Laplacian, and renders their sign patterns as binary segmentation masks.

## Algorithms

### Similarity (affinity matrix W)
Three methods for building the N×N pixel-similarity matrix `W_ij = exp(-(d_color²/σ_c² + d_space²/σ_s²))`:

| Method | Memory | Notes |
|--------|--------|-------|
| `dense` | O(N²) | Exact; default for small images |
| `knn` | O(N·k) | k-nearest-neighbour sparsification |
| `nystrom` | O(N·m) | Low-rank approximation via m landmark pixels |

### Eigensolver
Three solvers for the k smallest eigenpairs of the normalised Laplacian `L = I − D^{-½}WD^{-½}`:

| Method | Cost | Notes |
|--------|------|-------|
| `full` | O(N³) | Exact dense eigen; practical up to N ≈ 500 |
| `power_iter` | O(N²·k·t) | Power iteration with Gram-Schmidt deflation |
| `lanczos` | O(N²·q) | Krylov subspace; fast for large N, small k |

### ACA pipeline (matrix-free)
Avoids forming W explicitly. Builds a thin Cholesky factor A (r×N) via pivoted Adaptive Cross Approximation such that W ≈ AᵀA, then QR-decomposes the D-normalised factor and lifts eigenpairs via SVD. O(N·r) memory, O(N·r²) time — preferred for large images.

### Semi-supervised refinement
Optional Allen-Cahn energy minimisation on the spectral basis using convexity splitting. Provide a greyscale label image (`--u0`) with white=+1, black=−1, grey=unlabelled.

## Build

```bash
cargo build --release   # optimised
cargo build             # debug
```

## Run

```bash
./target/release/spectral_segment --input image.png --output out/
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-side N` | 64 | Downscale longest side to N before processing |
| `--k N` | 6 | Number of eigenvectors |
| `--pipeline classic\|aca` | classic | Algorithm variant |
| `--sim-method dense\|knn\|nystrom` | dense | Similarity construction |
| `--eigen-method full\|power_iter\|lanczos` | full | Eigensolver |
| `--sigma-color σ` | 0.1 | Colour bandwidth (HSL) |
| `--sigma-space σ` | 0.15 | Spatial bandwidth |
| `--u0 labels.png` | — | Label image for Allen-Cahn refinement |

Recommended settings by image size:

| `--max-side` | `--pipeline` | `--eigen-method` |
|---|---|---|
| ≤ 32 | classic | full |
| ≤ 64 | classic | lanczos |
| > 64 | aca | — |

Outputs written to `--output` directory: `eigvec_00.png` … `eigvec_k.png`, `summary.png`, `timing.json`.

## Test

```bash
cargo test                  # all tests
cargo test laplacian::      # tests for a specific module
cargo clippy                # lint
cargo fmt                   # format
```

## Benchmark

Sweeps eigensolver, similarity, and ACA rank parameters across synthetic images (gradient, checkerboard, noise) and reports eigenvalue errors, eigenvector residuals, and wall-clock timing.

```bash
cargo run --release --bin benchmark
```

## Examples

```
cargo run --release --bin spectral_segment \
-- \
--input examples/image.png \
--output out/ \
--max-side 128 \
--pipeline aca \
--num-eigenvectors 4 \
--sigma-color 0.015 \
--sigma-space 0.5 \
--aca-max-rank 15
```

## Benchmarks

See the [results](https://github.com/Farbdrucker/spectral-clustering-rs/blob/main/benchmark/macbook_air_m2_16gb.txt) of the benchmarks running on my local develop machine
