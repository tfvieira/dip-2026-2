# Task 11: Automatic Color Segmentation with K-Means

## Objective

Build an interactive color-segmentation pipeline that automatically groups the
chips in `chips.png` using K-means and circular Hue features.

## Description

This task continues Task 10. Previously, color groups were obtained with a
masked Hue histogram and manually selected Hue intervals. The background
removal step is provided. Students must use K-means to automatically group the
chip pixels by color.

The pipeline is:

```text
RGB image -> HSV -> Saturation mask -> Hue for chip pixels
-> circular Hue features -> K-means -> color-cluster segmentation
```

Hue is represented with cosine and sine components so that values near 0 and
360 degrees remain close in the clustering space. With `K = 5`, the clusters
should approximately correspond to the red, orange, yellow, green, and blue
chips. Cluster IDs are arbitrary and do not directly represent semantic color
names.

The notebook provides the RGB-to-HSV conversion, the `Saturation > 0.80` chip
mask, circular Hue conversion, deterministic K-means++ centroid initialization,
and all visualization code. Do not use `sklearn.cluster.KMeans`, `cv2.kmeans`,
SciPy clustering, or other ready-made clustering implementations.

## What Students Must Implement

Complete every marked region in the notebook:

- `assign_clusters(features, centroids)`;
- `update_centroids(features, labels, previous_centroids)`;
- `kmeans(features, k, max_iterations=100)`;
- `segment_chips_kmeans(hsv, chip_mask, k)`.

### K-means contract

- `features` has shape `(N, D)` and `centroids` has shape `(K, D)`.
- `assign_clusters` returns integer labels with shape `(N,)`, using the nearest
  centroid according to squared Euclidean distance.
- `update_centroids` returns the mean feature vector for each cluster. An empty
  cluster must keep its previous centroid and must not produce `NaN` values.
- `kmeans` must validate `1 <= k <= N`, use the supplied deterministic
  initializer, stop when labels do not change or when `max_iterations` is
  reached, and leave input arrays unchanged.

### Segmentation contract

- Use only `hsv[..., 0][chip_mask]` as the Hue data for clustering.
- Convert Hue values to the supplied circular features before calling K-means.
- The output label image has shape `(H, W)` and an integer dtype.
- Background pixels have label `0`; chip pixels have labels from `1` through
  `K`. Numeric cluster IDs are arbitrary.

## How to Run

Open `task-11-kmeans-color-segmentation.ipynb` in Jupyter or Google Colab and
run its cells from top to bottom. The first cell installs the toolkit and the
notebook downloads `chips.png`. Implement the four marked functions, run the
deterministic tests, and then use the `K` slider to compare `K = 3` through
`K = 7`. A correct solution prints `Test passed!` and displays the original
image, chip mask, K-means segmentation, and Hue histogram with cluster
representatives.
