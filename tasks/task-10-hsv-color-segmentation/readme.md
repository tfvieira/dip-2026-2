# Task 10: Interactive Color Segmentation in HSV

## Objective

Build an interactive HSV pipeline that segments the five colors of the course
chips image: red, orange, yellow, green, and blue.

## Description

The chips are highly saturated while the wooden background is weakly saturated.
The background must first be excluded using the Saturation channel. The Hue
histogram must then be computed using only pixels belonging to the chips. The
five Hue regions in that masked histogram can be used to define empirical,
deterministic color intervals.

The pipeline is:

```text
RGB image -> HSV -> Saturation mask -> masked Hue histogram
-> five Hue intervals -> five-class segmentation
```

This task uses empirical HSV analysis and manual Hue intervals. Automatic
clustering methods such as K-means are outside the scope of this task.

## What Students Must Implement

Complete every marked region in the notebook:

- `rgb_to_hsv(image)`;
- `create_saturation_mask(hsv, saturation_threshold)`;
- `compute_masked_hue_histogram(hsv, mask, num_bins=360)`;
- `segment_hue_ranges(hsv, chip_mask, hue_ranges)`.

### RGB to HSV contract

- Input is an RGB `uint8` array with shape `(H, W, 3)`.
- Output has the same shape and `float64` dtype.
- Hue is measured in degrees in `[0, 360)`; Saturation and Value are in
  `[0, 1]`.
- Do not modify the input or use `cv2.cvtColor`, scikit-image, or Matplotlib
  conversion helpers.

### Mask, histogram, and labels contract

- The chip mask is `hsv[..., 1] > saturation_threshold`, has dtype `bool`, and
  uses a threshold in `[0.0, 1.0]`.
- Hue histogram uses only `hsv[..., 0][mask]`, defaults to 360 bins over
  `[0, 360)`, and must exclude background pixels.
- Labels are `0` for background, then `1` through `5` for red, orange, yellow,
  green, and blue, in that order.
- Hue ranges must support a wrapped interval, so red can include values near
  both `0` and `360` degrees.

Do not use K-means, clustering, morphology, connected components, individual
chip counting, automatic white balance, HSV/LAB conversion helpers, or
`cv2.inRange`.

## How to Run

Open `task-10-hsv-color-segmentation.ipynb` in Jupyter or Google Colab and run
the cells from top to bottom. The notebook downloads `chips.png` using
`download_course_image`, provides an interactive Saturation slider, and leaves
the four core functions for implementation. A correct solution prints
`Test passed!` and displays the HSV channels, chip mask, masked Hue histogram,
and final five-class segmentation.
