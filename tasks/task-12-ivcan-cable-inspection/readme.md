# Task 12: IVCAN Cable Inspection

## Objective

Develop and evaluate methods for segmenting automotive cables and their color stripes using the IVCAN/Prysmian dataset.

This task is designed for both Digital Image Processing (DIP) and Deep Learning (DL) students. The goal is to implement segmentation methods, evaluate their performance across five fixed folds, and analyze the results using segmentation metrics.

## Description

Complete the four functions marked with `TODO` in the provided notebook:

1. **`segmenta_cabo(imagem_rgb, parametros=None)`**  
   Implement a method to identify the cable in an RGB image and return its binary mask.

2. **`segmenta_listra(imagem_rgb, mascara_cabo_predita, parametros=None)`**  
   Implement stripe segmentation using the RGB image and the predicted cable mask.

3. **`separar_amostras_fold(fold_por_amostra, fold_teste)`**  
   Separate training and testing samples for a selected fold, keeping all other folds available for training.

4. **`resumir_cinco_folds(por_fold)`**  
   Calculate the mean and sample standard deviation (`ddof=1`) of the results obtained across the five folds.

The notebook already provides dataset loading, ground-truth masks, fixed fold assignments, segmentation metrics, visualization tools, contract validation, and the main evaluation infrastructure.

Students are responsible for completing the marked functions without modifying the provided evaluation logic.

DIP students may explore color thresholds, HSV/Lab color spaces, morphology, K-means, or other image-processing techniques. DL students may use neural networks, provided they follow the same function contracts and evaluation protocol.

### Segmentation contract

- Input images are RGB. When using OpenCV, apply explicit conversions such as `cv2.COLOR_RGB2HSV` or `cv2.COLOR_RGB2LAB`.
- Each segmentation function must return a binary 2D mask with the same height and width as the input image.
- Accepted mask values are boolean, `0/1`, or `0/255`.
- The cable mask must include the stripe region.
- The stripe mask may be entirely empty when no stripe is visible.
- `segmenta_listra` must use the **predicted cable mask**, never the ground-truth cable mask.
- Do not access sample IDs, reference masks, ground truth, or external labels inside the segmentation functions.
- Preserve the provided function signatures.
- Do not modify the fixed fold assignments, provided metrics, or evaluation infrastructure.

## Dataset Setup in Google Colab

The dataset is available as a public ZIP file:

[Download IVCAN-Dataset.zip](https://drive.google.com/file/d/1q0OG-pHmEKcgAML49jlaZTmn0suO1p2j/view)

The notebook automatically downloads and extracts the ZIP using `gdown`. No Google Drive mount, shortcut, or manual dataset configuration is required.

The extracted dataset contains:

- `dataset.csv`: sample metadata;
- `Images/`: 80 folders containing 320 images;
- `Masks/`: 80 folders containing 552 mask files.

The setup cell validates the dataset structure before continuing.

Downloaded files are stored temporarily in the Colab environment. Evaluation results are saved separately in `/content/ivcan_results`.

**Note:** The Colab filesystem is temporary. Downloaded data and generated results will be deleted when the runtime is reset. Save any results you want to keep before closing the session.

## Five-Fold Evaluation Protocol

The dataset is divided into five fixed folds. These assignments must not be modified.

For each test fold `k`, the remaining four folds must be used for training and parameter selection. Parameters must be recorded in `parametros_por_fold` and fixed before evaluating the corresponding test fold.

The evaluator does not train models or automatically optimize parameters. Any training or parameter tuning must be performed separately, using only the appropriate training samples.

The four cameras of each ID-experiment pair remain in the same fold. IDs 018, 028, 031, and 040 have separate experiments; this preserves examples of exclusive colors in training but evaluates different experiments of the same physical cable. This limitation must be discussed in the report.

### Evaluation and statistics

Students must implement the train/test separation and statistical summary functions.

The notebook provides deterministic tests to verify both implementations. These tests must pass before completing the full evaluation.

The evaluation follows these rules:

- Segmentation metrics are calculated separately for cable and stripe masks.
- Metrics are calculated per image and macro-averaged within each fold.
- The final mean gives equal weight to all five folds.
- The sample standard deviation is calculated across folds using `ddof=1`.
- Undefined metrics remain `NaN` and are excluded from the corresponding means.
- Valid-value counts are reported to clarify how many observations contributed to each statistic.

The final results must include the metrics for each fold, their means and standard deviations, and an analysis of the observed segmentation errors.

### Dataset limitations

A missing stripe-mask file is interpreted as an empty stripe ground-truth mask. This convention should be considered when interpreting the results, since a missing annotation may not necessarily indicate the absence of a stripe.

When `ignorar_erros_dados=True`, the evaluator records and excludes images with identified data problems.

All exclusions must be documented. Do not present a partial evaluation as if all 320 theoretical images had been processed.

## How to Run

1. Open `task-12-ivcan-cable-inspection.ipynb` in Google Colab.
2. Run the initial cells to download and validate the public dataset.
3. Inspect the sample images, reference masks, and provided metrics.
4. Implement `segmenta_cabo` and `segmenta_listra` in the marked `TODO` regions.
5. Implement `separar_amostras_fold` and run its deterministic test.
6. Configure and tune the segmentation parameters using training samples only.
7. Run the segmentation contract test and inspect the predicted masks.
8. Implement `resumir_cinco_folds` and run its deterministic test.
9. Execute the complete five-fold evaluation.
10. Analyze the results and discuss the methods, metrics, limitations, and possible improvements.

## Expected Deliverables

Submit the completed notebook containing:

- The four implemented functions, with all `TODO` regions completed.
- The segmentation method and parameters used for each fold.
- Successful execution of the provided deterministic and contract tests.
- Segmentation results for cable and stripe detection.
- Evaluation tables covering all five folds, including means and sample standard deviations.
- A discussion of segmentation errors, dataset exclusions, limitations, and possible improvements.

The objective is not only to execute the evaluation, but to develop and justify a segmentation strategy and analyze how well it performs on the IVCAN dataset.