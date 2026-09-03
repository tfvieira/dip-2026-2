# Task 05 — Sobel Gradient

## Objective

Implement manual Sobel edge detection for a grayscale image using only NumPy.
The exercise develops the connection between local intensity changes and image
edges.

## Description

An image gradient describes how its intensity changes from one pixel location
to another. The Sobel operator uses two 3x3 kernels: one produces `Gx`, which
measures horizontal intensity variation, and the other produces `Gy`, which
measures vertical intensity variation. The gradient magnitude combines both
responses to represent the strength of the local change.

Apply the kernels as **correlation**, without flipping them. Use zero padding:
pixels outside the image boundaries must be considered equal to `0`, and the
outputs must preserve the input shape.

Use only NumPy. Ready-made Sobel, correlation, convolution, or edge-detection
functions are not allowed, including `cv2.Sobel`, `cv2.filter2D`, `cv2.Canny`,
`scipy.ndimage.sobel`, `scipy.signal.correlate2d`, and
`scipy.signal.convolve2d`.

## What Students Must Implement

Complete `sobel_gradient` in `task-05-sobel-gradient.py`. The input is a
non-empty two-dimensional grayscale NumPy array with dtype `uint8`.

The function must return `gradient_x`, `gradient_y`, and `magnitude`. Each
result must be a new `float64` array with the same shape as the input, and the
input image must not be modified.

Use exactly these kernels:

```text
Sobel X                 Sobel Y
[-1,  0, 1]             [-1, -2, -1]
[-2,  0, 2]             [ 0,  0,  0]
[-1,  0, 1]             [ 1,  2,  1]
```

After calculating both responses, calculate the magnitude as
`sqrt(Gx² + Gy²)`. Do not normalize or clip the magnitude.

Only edit the code between `### START CODE HERE ###` and
`### END CODE HERE ###`.

## How to Run

After completing the function, run:

```bash
python task-05-sobel-gradient.py
```

The script checks a vertical edge, a zero-valued image, output shape and dtype,
and input immutability. A correct implementation prints:

```text
Test passed!
```
