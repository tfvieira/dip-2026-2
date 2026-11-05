# Task 13: Morphological Operations

## Objective

Implement the four fundamental binary morphological operations manually with
NumPy: dilation, erosion, opening, and closing.

## Description

This task uses 2D binary images, where `0` denotes background and `255` denotes
foreground. A structuring element is also a binary 2D array, where `1` marks an
active position. All operations must preserve the input image shape.

The notebook provides square structuring elements of size 3x3 and 5x5, plus a
cross-shaped element. The element dimensions are always odd, its anchor is the
center pixel, and pixels outside the image boundary are treated as background
(`0`).

## What Students Must Implement

Complete the four marked regions in the notebook:

- `dilatacao_binaria(imagem, elemento_estruturante)`;
- `erosao_binaria(imagem, elemento_estruturante)`;
- `abertura_binaria(imagem, elemento_estruturante)`;
- `fechamento_binario(imagem, elemento_estruturante)`.

### Operation contracts

- Inputs and outputs are 2D NumPy arrays with the same image dimensions.
- Output masks must contain only `0` and `255`.
- Inputs must not be modified.
- Dilation expands foreground according to active structuring-element positions.
- Erosion retains a foreground pixel only when every active position is also
  foreground.
- Opening must reuse the student erosion followed by dilation.
- Closing must reuse the student dilation followed by erosion.

Do not call `cv2.dilate`, `cv2.erode`, `cv2.morphologyEx`, SciPy/scikit-image
morphology functions, or any ready-made equivalent inside the four functions.
Use NumPy, loops, and pixel neighborhoods to implement the logic manually.

## How to Run

Open `task-13-morphological-operations.ipynb` in Jupyter or Google Colab. Run
the cells from top to bottom, implement the four TODO regions, and then run the
deterministic tests. A correct solution prints `Test passed!` and displays a
comparison between the original image, dilation, erosion, opening, and closing.

Experiment with the 3x3 square, 5x5 square, and cross structuring elements, and
answer the discussion questions at the end of the notebook.
