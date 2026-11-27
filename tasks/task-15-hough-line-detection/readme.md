# Task 15: Hough Transform for Line Detection

## Objective

Build a pipeline that detects image edges, finds straight lines with the
standard Hough transform, then clips and draws the lines on an RGB image.

## What Students Must Implement

- `detectar_bordas`: RGB to grayscale, Gaussian smoothing and Canny.
- `detectar_retas`: standard OpenCV Hough transform with floating-point
  `(N, 2)` output, including `(0, 2)` for no detections.
- `calcular_extremos_reta`: convert `(rho, theta)` to integer endpoints
  within the image, or return `None` for an invisible line.
- `desenhar_retas`: draw visible lines on an independent RGB copy.

RGB images are three-channel uint8 arrays. Grayscale images and edge maps
are 2D uint8 arrays, with edges encoded as 255 and background as 0.
Line angles are in radians. Preserve inputs and image dimensions.

## How to Run

Open [the notebook](task-15-hough-line-detection.ipynb) in Jupyter or Google
Colab. For a local kernel, install NumPy, Matplotlib and OpenCV
(`pip install numpy matplotlib opencv-python`). All images are generated
in the notebook.

Complete the four TODOs and run the notebook in order. A correct solution
prints `Test passed!`. Compare at least three parameter configurations on
the same image and discuss detection counts, false positives and missed lines.
