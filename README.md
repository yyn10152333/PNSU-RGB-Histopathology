# H&E spectral decomposition and color normalization

This repository contains the MATLAB code needed to run the H&E joint spectral and concentration estimation method used in the paper.

The package is intentionally minimal. It includes the solver, its helper functions, and the fitted spectral initialization files required at runtime. Synthetic images, ground truth, evaluation scripts, and previously generated experiment outputs are not included.

## Contents

- `he_spectral_decomposition.m`: main solver using the manuscript-aligned implementation.
- `run_example.m`: small wrapper for running the solver with the paper experiment settings.
- `solve_concentration_fast_GN_patch*.m`: concentration fitting helpers.
- `Gaussian_reconstruction5.m`, `rgb2od.m`, `od2rgb.m`: spectral reconstruction and RGB/OD conversion helpers.
- `p_fit5_result_*.mat`: fitted spectra used to initialize H, E, and RGB response curves.

## Requirements

- MATLAB R2023b or a nearby release
- Optimization Toolbox
- Parallel Computing Toolbox
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

## Quick Start

In MATLAB, add this directory to the path and call:

```matlab
run_example(referenceImage, inputImageDirectory, outputDirectory)
```

Example:

```matlab
cd('path/to/this/repository')
run_example( ...
    'path/to/reference.png', ...
    'path/to/input_images', ...
    'path/to/output')
```

The input directory may contain `.png`, `.jpg`, `.jpeg`, `.tif`, or `.tiff` images. The wrapper uses the same settings as the paper experiments:

```matlab
C_list = 100;
P_list = 2;
NumSample = 600;
ThrWhite = 240;
LowCut = 15;
lambda_orthoC = 0;
lambda_s = 0;
PatchVis = false;
```

## Outputs

Results are written to:

```text
<outputDirectory>/path0/<image-name>/
```

Each image directory contains:

- `results_data.mat`: estimated spectra and concentration maps.
- `converge_hist.mat`: optimization history.
- `01_RGB_orig.png`: original image.
- `02_RGB_recon_self.png`: self-reconstructed RGB image.
- `03_RGB_diff.png`: reconstruction difference visualization.
- `04_RGB_corrected_ref.png`: reference-normalized RGB image.
- `Hematoxylin.png` and `Eosin.png`: separated H/E component images.

The solver also creates a local `ref/` directory on the first run. This cache stores spectra and concentration ranges estimated from the reference image. Delete `ref/` if you intentionally want to recompute the reference state from a different reference image.

## Reproducibility Notes

- Use `run_example.m` when possible. It changes into the code directory before running, so the initialization `.mat` files are resolved consistently.
- The solver uses a fixed random seed for pixel sampling.
- For a fair comparison across runs, keep the same reference image, input image directory, and `ref/` cache state.
- This release contains the updated manuscript-aligned implementation. It is expected to be close in trend to the historical experiment code, but not numerically identical.
