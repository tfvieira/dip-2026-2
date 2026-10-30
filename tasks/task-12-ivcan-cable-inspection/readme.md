# Task 12: IVCAN Cable Inspection

## Objective

Develop and evaluate a method for segmenting automotive cables and their color
stripes in the IVCAN/Prysmian dataset. The same task supports both Digital
Image Processing (DIP) and Deep Learning (DL) approaches.

## Description

Implement the internal logic of the two functions already provided in the
notebook:

- `segmenta_cabo(imagem_rgb, parametros=None)`;
- `segmenta_listra(imagem_rgb, mascara_cabo_predita, parametros=None)`.

The notebook provides data reading, ground-truth masks, fixed five-fold splits,
metrics, contract checks, and the final evaluator. Preserve the function
signatures and use only the input arguments inside each segmentation function.
`segmenta_listra` receives the **predicted** cable mask, never the real one.

DIP students may use color rules, thresholds, morphology, or optional K-means.
DL students may replace the internal strategy with a neural model. Both routes
must respect the same contracts and five-fold protocol for a fair comparison.

### Segmentation contract

- Inputs are RGB images. If using OpenCV, use explicit RGB conversions such as
  `cv2.COLOR_RGB2HSV` or `cv2.COLOR_RGB2LAB`.
- Return a binary 2D mask with the input image height and width. Accepted values
  are boolean, `0/1`, or `0/255`.
- The cable mask includes the stripe region; stripe segmentation may return an
  empty mask when no stripe is visible.
- Do not access sample IDs, reference masks, ground truth, or external labels
  inside the segmentation functions.
- Do not modify the provided folds, metrics, evaluator, or function signatures.


## Dataset Setup in Google Colab

The dataset is provided as a public ZIP file:

[Download IVCAN-Dataset.zip](https://drive.google.com/file/d/1q0OG-pHmEKcgAML49jlaZTmn0suO1p2j/view)

The notebook automatically downloads and extracts the ZIP using
`gdown`. No Google Drive mount, shortcut, or manual dataset
configuration is required.

The extracted dataset contains:

- `dataset.csv`: sample metadata;
- `Images/`: 80 folders containing 320 images;
- `Masks/`: 80 folders containing 552 mask files.

The setup cell validates the required files and directories before
continuing.

Downloaded files are stored temporarily in the Colab environment.
Evaluation results are saved separately in `/content/ivcan_results`.

**Note:** The Colab filesystem is temporary. Downloaded data and
results will be deleted when the runtime is reset. Save any results
you want to keep before closing the session.


## Five-Fold Evaluation Protocol

The five folds are fixed. For test fold `k`, choose parameters only with the
other folds, record them in `parametros_por_fold`, and freeze them before test.
The evaluator does not train a model or search parameters.

The four cameras of each ID-experiment pair stay in the same fold. IDs 018, 028,
031, and 040 have separate experiments; this preserves exclusive colors in
training but evaluates different experiments of the same physical cable. State
this limitation in the report.

Metrics are calculated per image and macro-averaged inside each fold. The final
mean gives equal weight to all five folds, while the sample standard deviation
uses `ddof=1`. Undefined metrics remain `NaN`; the notebook reports valid-value
counts in each summary.

A missing stripe-mask file means no stripe in the ground truth. Confirm that it
is an annotation decision rather than a missing label. If
`ignorar_erros_dados=True`, document every exclusion and do not report a partial
evaluation as if all 320 theoretical images had been processed.

## How to Run

Open `task-12-ivcan-cable-inspection.ipynb` in Google Colab. Download and
validate the public dataset, inspect an example, implement only the two marked
regions, choose and freeze parameters per fold, run the contract test, and then
execute the five-fold evaluation. Report the cable and stripe tables, fold
parameters, data exclusions, and an analysis of errors and limitations.
