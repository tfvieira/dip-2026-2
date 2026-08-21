# Task 04 — Spatial Correlation

## Objective

Implement two-dimensional spatial correlation for a grayscale image using
only NumPy. The goal is to understand how a kernel is applied to each local
neighborhood of an image.

## Description

The input image is a two-dimensional grayscale NumPy array. For every pixel,
apply a two-dimensional kernel to the corresponding neighborhood and store the
sum of the element-wise products in the output.

Use **zero padding** outside the image boundaries so that the output has the
same shape as the input image. Pixels outside the original image must be
considered equal to `0`.

This task requires **correlation**, not convolution. In correlation, the
kernel is applied exactly as provided. Unlike convolution, the kernel is not
flipped horizontally or vertically.

Use only NumPy. Ready-made correlation or convolution functions are not
allowed, including `cv2.filter2D`, `scipy.signal.correlate2d`,
`scipy.signal.convolve2d`, `np.convolve`, or equivalent functions from other
libraries.

## What Students Must Implement

Complete the `correlate2d` function in
`task-04-spatial-correlation.py`. The input `image` is a non-empty grayscale
array with shape `(height, width)`. The input `kernel` is a non-empty 2D array
with odd height and width.

The function must return a new NumPy array with the same shape as `image` and
dtype `float64`. It must not modify either `image` or `kernel`.

Only edit the code between `### START CODE HERE ###` and
`### END CODE HERE ###`. The asymmetric kernel in the test case is intentional:
it detects an incorrect convolution-style implementation that flips the
kernel.

## How to Run

After completing the function, run:

```bash
python task-04-spatial-correlation.py
```

The script verifies the asymmetric-kernel result, the identity kernel, output
shape and dtype, and input immutability. A correct implementation prints:

```text
Test passed!
```
