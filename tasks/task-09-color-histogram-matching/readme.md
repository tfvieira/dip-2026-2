# Task 09: Reference-Based Color Standardization with Histogram Matching

## Objective

Build an RGB color-standardization pipeline that matches the channel
distributions of a source image to the corresponding distributions of a
reference image.

## Description

Images of the same scene may have different color casts when acquired under
different conditions. In this task, a warmer reference image and a bluer source
image are generated from the same synthetic scene. You will match the source
histogram to the reference histogram independently for the red, green, and blue
channels.

For each source intensity `i`, the lookup table must select the smallest
reference intensity `j` such that:

```
reference_cdf[j] >= source_cdf[i]
```

The goal is not to independently equalize the source image. The source RGB
channel distributions must be matched to the corresponding distributions of a
reference image.

## What Students Must Implement

Complete every marked region in the notebook:

- `compute_histogram_cdf(channel)`;
- `match_channel(source_channel, reference_channel)`;
- `match_color_histograms(source, reference)`.

### Histogram and CDF contract

- Input is a two-dimensional `uint8` channel.
- Histogram has exactly 256 bins for intensities `0` through `255`.
- CDF has 256 `float64` values, remains in `[0.0, 1.0]`, and ends in `1.0`.
- `np.bincount(channel.ravel(), minlength=256)` may be used.

### Matching contract

- Source and reference channels are `uint8`; the output has the source shape
  and dtype `uint8`.
- Build one deterministic 256-element LUT from the two CDFs, then apply it to
  the source channel. `np.searchsorted` may be used.
- Color inputs are RGB arrays with shape `(H, W, 3)` and dtype `uint8`.
- Source and reference spatial dimensions may differ.
- Process the channels independently: `R -> R`, `G -> G`, and `B -> B`.
- Do not modify either input and return a new array.

Do not use `skimage.exposure.match_histograms`, ready-made color
standardization, RGB 3D joint matching, CLAHE, Gray World, Retinex, HSV, LAB,
or automatic white balance functions.

## How to Run

Open `task-09-color-histogram-matching.ipynb` in Jupyter or Google Colab and
run the cells from top to bottom. Implement the three marked functions before
running the deterministic checks. A correct solution prints `Test passed!` and
then visualizes the reference, source, and matched images alongside their RGB
histograms and CDFs.
