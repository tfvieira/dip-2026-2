# Task 06: White Balance

## Objective

Implement Gray World white balance for an RGB image using only NumPy. The goal
is to understand how a color cast can be reduced by balancing the average
intensity of the three color channels.

## Description

Different illumination sources can introduce a color cast into an image. White
balance compensates for that cast so neutral scene content is represented more
consistently. The Gray World method assumes that, on average, a scene should
have similar intensity in the R, G, and B channels.

Calculate the mean of each RGB channel and use them to compute the Gray World
reference mean:

`gray_mean = (mean_R + mean_G + mean_B) / 3`

Then calculate one correction gain per channel:

`gain_R = gray_mean / mean_R`  
`gain_G = gray_mean / mean_G`  
`gain_B = gray_mean / mean_B`

Apply each gain to its corresponding channel. Round the corrected values to
the nearest integer, clip them to `[0, 255]`, and return a new `uint8` image.

If a channel has mean zero, use a gain of `1.0` for that channel to avoid
division by zero.

This task uses the explicit **RGB** channel order:

- `image[:, :, 0]` is Red;
- `image[:, :, 1]` is Green;
- `image[:, :, 2]` is Blue.

Use only NumPy. Ready-made white-balance functions are not allowed, including
`cv2.xphoto.createGrayworldWB`, `cv2.xphoto.createSimpleWB`, or equivalent
functions from other libraries.

## What Students Must Implement

Complete `white_balance` in `task-06-white-balance.py`. The input is a
non-empty NumPy array with shape `(height, width, 3)`, dtype `uint8`, and RGB
channel order.

The function must return a new array with the same shape and dtype. Its values
must remain in `[0, 255]`, and the input image must not be modified. Only edit
the code between `### START CODE HERE ###` and `### END CODE HERE ###`.

## How to Run

After completing the function, run:

```bash
python task-06-white-balance.py
```

The script checks a color-cast example, an already neutral image, a channel
whose mean is zero, output shape and dtype, value range, and input
immutability. A correct implementation prints `Test passed!`.
