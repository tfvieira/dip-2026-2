# Task 03 — Histogram Equalization

## Objective

The goal of this exercise is to implement histogram equalization for a
grayscale image. Histogram equalization remaps pixel intensities to improve
the distribution of contrast in an image.

## Description

The input is a two-dimensional grayscale NumPy array with dtype `uint8`.
Students must calculate the histogram with 256 intensity levels, compute the
cumulative distribution function (CDF), and use it to build a new intensity
mapping.

The implementation must use only NumPy. Ready-made histogram equalization
functions, such as `cv2.equalizeHist`, are not allowed.

## What Students Must Implement

Students are required to complete the `equalize_histogram` function by:

- calculating the histogram of the input image;
- calculating the CDF and its first nonzero value;
- building an equalization lookup table;
- applying the lookup table to the image.

For an input intensity `r`, use the following mapping convention:

```text
s = round((CDF(r) - cdf_min) / (N - cdf_min) * 255)
```

In this expression, `CDF(r)` is the cumulative count at intensity `r`,
`cdf_min` is the first nonzero CDF value, and `N` is the total number of
pixels in the image. This mapping defines the values used in the equalization
lookup table.

The function includes markers (`### START CODE HERE ###` and
`### END CODE HERE ###`). Only the code between these markers should be
modified.

For a constant image, the function must return an equivalent copy of the
input.

## How to Run

After completing the function, run the script to execute the test cases:

```bash
python task-03-histogram-equalization.py
```

The tests verify the equalized result, shape, dtype, input immutability, and
the behavior for a constant image. A success message `Test passed!` confirms
correctness.
