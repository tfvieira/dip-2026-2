# Task 14: Thinning and Skeletonization

## Objective

Implement manual binary thinning with the Zhang-Suen algorithm and morphological
skeletonization with NumPy. This task continues the morphology study from Task
13 by reducing objects to thinner structural representations.

## Description

Images are 2D binary arrays: `0` is background and `255` is foreground. Outputs
must preserve input dimensions, use binary values, and never modify their input
arrays. Zhang-Suen uses 8-neighbor connectivity; pixels outside image boundaries
are background.

The notebook presents Zhang-Suen thinning and morphological skeletonization as
different methods. Their outputs do not need to be identical: thinning focuses
on connectivity-preserving width reduction, while a morphological skeleton is
constructed from successive erosions and openings.

## What Students Must Implement

Complete the four marked regions in the notebook:

- `contar_transicoes_01(vizinhos)`;
- `passo_afinamento(imagem, etapa)`;
- `afinamento_zhang_suen(imagem)`;
- `esqueletonizacao_morfologica(imagem, elemento_estruturante)`.

### Contracts

- The neighbor sequence is circular and ordered as `P2, P3, ..., P9` around
  the center pixel.
- `passo_afinamento` must implement one Zhang-Suen subiteration, mark removals,
  and apply them simultaneously.
- Full thinning repeats both subiterations until neither removes a pixel.
- Morphological skeletonization must accumulate the difference between each
  eroded image and its morphological opening until erosion is empty.
- The supplied helper erosion, dilation, and opening functions may be reused
  only for morphological skeletonization.

Do not use ready-made thinning or skeletonization functions from OpenCV, SciPy,
or scikit-image inside the TODO functions.

## How to Run

Open `task-14-thinning-skeletonization.ipynb` in Jupyter or Google Colab. Run
the cells in order, implement all TODO regions, and run the deterministic tests.
A correct solution prints `Test passed!` and displays the original synthetic
objects next to Zhang-Suen thinning and morphological skeletonization results.
