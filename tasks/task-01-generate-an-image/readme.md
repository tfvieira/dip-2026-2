# Gaussian Image Generator - Implementation Exercise

## Objective

The goal of this exercise is to implement a function that generates a grayscale image with pixel values sampled from a normal (Gaussian) distribution. The function must be reproducible based on a random seed provided as input. This helps students understand how to control random number generation in image processing and how distributions influence pixel intensity patterns.

## Description

The image is represented as a NumPy array of shape `(height, width)` with float values. Each pixel is sampled from a normal distribution defined by the given `mean` and `std` (standard deviation). The use of a random seed ensures that the generated image is consistent across different executions when the same parameters are used.

The exercise includes:

- Initialization of a NumPy random generator with a fixed seed
- Use of `numpy.random.normal` to generate values from a Gaussian distribution
- Creation of a 2D array simulating a grayscale image
- Verification using a fixed test case with expected pixel values

## What Students Must Implement

Students are required to complete the `generate_image` function by:

- Setting the NumPy random seed
- Generating the image array

The function includes markers (`### START CODE HERE ###` and `### END CODE HERE ###`) and a TODO comment to indicate where the implementation is expected.

## How to Run

After completing the function, run the script to execute a test case:

- The test checks if the generated image closely matches a predefined matrix
- The comparison allows for a small floating-point tolerance
- A success message "Test passed!" confirms correctness

```bash
python task-01-generate-an-image.py
```
