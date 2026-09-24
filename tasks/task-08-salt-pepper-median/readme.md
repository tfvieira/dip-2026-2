# Task 08: Interactive Salt-and-Pepper Noise Removal

## Objective

Build an interactive grayscale-image pipeline that adds deterministic
salt-and-pepper noise and removes it with a manually implemented median filter.

## Description

Salt-and-pepper noise replaces selected pixels with the minimum intensity
(``0``, pepper) or the maximum intensity (``255``, salt). A median filter
replaces each pixel by the median of a local neighborhood and is especially
useful for this impulsive noise. In this task, you will explore the trade-off
between noise density and the odd window size used by the filter.

The notebook compares the noisy and restored images visually and reports RMSE:

```
sqrt(mean((reference - result) ** 2))
```

## What Students Must Implement

Complete every marked region in the notebook:

- `add_salt_pepper_noise(image, density, seed=0)`;
- `median_filter(image, kernel_size)`;
- `rmse(reference, result)`;
- the `kernel_size_slider`, using the provided density slider as a reference.

### Noise contract

- Input: grayscale `uint8` image and a density in `[0.0, 1.0]`.
- Use `np.random.default_rng(seed)` so a fixed seed is reproducible.
- Altered pixels are split between pepper (`0`) and salt (`255`).
- Return a new `uint8` array without changing the input.

### Median-filter contract

- Input: grayscale `uint8` image.
- `kernel_size` is odd and at least `3`.
- Use `np.pad(..., mode="reflect")` for borders.
- For every pixel, calculate `np.median` over its `kernel_size x kernel_size`
  neighborhood manually.
- Return a new `uint8` array with the input shape and without changing input.

Do not use ready-made noise generators or filters, including
`cv2.medianBlur`, `scipy.ndimage.median_filter`, or scikit-image filters.
Do not add Gaussian, mean, adaptive, RGB, or Fourier processing.

## How to Run

Open `task-08-salt-pepper-median.ipynb` in Jupyter or Google Colab. Complete
the implementation cells, run the deterministic test until it prints
`Test passed!`, then create the missing window-size slider and explore the
pipeline.

The notebook provides the density slider with `continuous_update=True`. Create
a `widgets.SelectionSlider` for the window size with the odd options `3`, `5`,
`7`, and `9`, also using `continuous_update=True`.
