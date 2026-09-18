# Task 07: Interactive Periodic Noise Removal with Fourier

## Objective

Build an interactive frequency-domain pipeline that adds sinusoidal noise to a
grayscale image and removes it with an ideal notch-reject filter.

## Description

Periodic or sinusoidal noise is a repeating intensity pattern. A spatial
sine wave produces specific, symmetric impulses in the frequency spectrum.
`np.fft.fft2` computes the 2D DFT and `np.fft.fftshift` centers its spectrum
for inspection. An ideal notch-reject filter removes selected frequency
components; `np.fft.ifftshift` and `np.fft.ifft2` reconstruct the result.

The notebook uses RMSE to compare the noisy and restored images with the
original. Sliders allow you to explore amplitude, horizontal and vertical
frequencies, and notch radius interactively.

Conceptual references:

- [2D Discrete Fourier Transform](https://www.youtube.com/watch?v=uD2BerBmnUs)
- [Adding periodic noise](https://www.youtube.com/watch?v=df3c53Pu09c)

## What Students Must Implement

Complete the marked regions in the notebook:

- create a 2D sinusoidal noise pattern;
- add it to a grayscale image without clipping;
- construct an ideal notch-reject filter in shifted-frequency coordinates;
- restore the image using FFT, shift, filter multiplication, inverse shift,
  inverse FFT, and the real part;
- calculate RMSE in `float64`;
- create sliders and connect them to the interactive pipeline.

Use `np.fft.fft2`, `np.fft.fftshift`, `np.fft.ifftshift`, and `np.fft.ifft2`.
Do not use OpenCV DFT/IDFT, SciPy signal functions, ready-made notch filters,
or automatic periodic-noise removal. Keep all calculations grayscale and do
not clip or convert the noisy/restored images to `uint8`.

## How to Run

Open `task-07-periodic-noise-fourier.ipynb` in Jupyter or Google Colab and run
the cells from top to bottom. Complete each `START CODE HERE` region before
running the deterministic test and the interactive experiment.

The test checks the known sinusoid, symmetric notch positions, reconstruction,
dtype, shape, input immutability, and RMSE improvement. A correct solution
prints `Test passed!`.
