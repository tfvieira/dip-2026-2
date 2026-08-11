# Salt and Pepper Image Generator - Implementation Exercise

## Objective

The goal of this exercise is to implement a function that generates a grayscale image with salt-and-pepper noise. This helps students understand how to simulate noise in image processing and how to manipulate probability distributions to introduce artifacts in digital images.

## Description

The image is represented as a NumPy array of shape `(height, width)` with float values. Each pixel in the image may take one of the following values:

- **1.0** (salt)
- **-1.0** (pepper)
- **0.5** (no noise)

The assignment uses two probabilities:

- `salt_prob`: probability that a pixel becomes salt (1.0)
- `pepper_prob`: probability that a pixel becomes pepper (-1.0)

All other pixels remain at 0.5. The function must generate this image using NumPy's `random.rand`.

## What Students Must Implement

Students are required to complete the `create_salt_and_pepper_noise` function by:

- Initializing a base image with value 0.5
- Generating a 2D noise matrix
- Assigning pixel values according to the salt and pepper probabilities

Markers (`### START CODE HERE ###` and `### END CODE HERE ###`) indicate where the implementation should be written.

## How to Run

After completing the function, run the script to execute a test case:

- The test checks if the number of salt and pepper pixels is approximately correct.
- The acceptable range allows for slight variation due to randomness.
- A success message "Test passed!" confirms correctness.

```bash
python task-02-salt-and-pepper.py
```
