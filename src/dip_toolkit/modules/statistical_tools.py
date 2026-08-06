from __future__ import annotations

from numbers import Real

import cv2 as cv
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


class StatisticalTools:
    def __init__(self):
        pass

    def get_image_info(
        self,
        image: np.ndarray,
        *,
        expected_range: tuple[Real, Real] | None = None,
    ) -> dict[str, object]:
        """Retorna propriedades estruturadas de uma imagem NumPy.

        Aceita imagens grayscale 2D e imagens coloridas 3D com três canais. Para
        dtypes inteiros, ``expected_range`` é o intervalo completo do dtype. Para
        floats, o padrão é o intervalo representável pelo dtype, sem supor que a
        imagem esteja em ``[0.0, 1.0]``. Use ``expected_range`` para declarar uma
        faixa didática específica, como ``(-1.0, 1.0)``.

        Args:
            image: Array NumPy 2D ou 3D, não vazio, com dtype inteiro ou float.
            expected_range: Faixa opcional ``(mínimo, máximo)`` esperada para os
                valores da imagem.

        Returns:
            Dicionário com dimensões, canais, shape, dtype, bytes, extremos,
            faixa esperada e estatísticas resumidas.
        """

        self._validate_image(image)
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        minimum = image.min().item()
        maximum = image.max().item()
        resolved_range = self._resolve_expected_range(image.dtype, expected_range)

        return {
            "width": width,
            "height": height,
            "ndim": image.ndim,
            "channels": channels,
            "shape": image.shape,
            "dtype": image.dtype,
            "nbytes": image.nbytes,
            "minimum": minimum,
            "maximum": maximum,
            "expected_range": resolved_range,
            # Chaves legadas preservadas para os exemplos já existentes.
            "depth": channels,
            "min_value": minimum,
            "max_value": maximum,
            "mean": float(np.mean(image)),
            "std_dev": float(np.std(image)),
        }

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image deve ser um array NumPy.")
        if image.ndim not in {2, 3}:
            raise ValueError("image deve ter duas ou três dimensões.")
        if any(dimension <= 0 for dimension in image.shape):
            raise ValueError("image não pode possuir dimensões vazias.")
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError("Imagens coloridas devem possuir exatamente três canais.")
        if not (
            np.issubdtype(image.dtype, np.integer)
            or np.issubdtype(image.dtype, np.floating)
        ):
            raise TypeError("image deve usar um dtype NumPy inteiro ou float.")

    @staticmethod
    def _resolve_expected_range(
        dtype: np.dtype,
        expected_range: tuple[Real, Real] | None,
    ) -> tuple[int | float, int | float]:
        if expected_range is not None:
            if not isinstance(expected_range, tuple) or len(expected_range) != 2:
                raise TypeError("expected_range deve ser uma tupla (mínimo, máximo).")
            lower, upper = expected_range
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                for value in expected_range
            ):
                raise TypeError("Os limites de expected_range devem ser números reais.")
            lower_float, upper_float = float(lower), float(upper)
            if not np.isfinite(lower_float) or not np.isfinite(upper_float):
                raise ValueError("Os limites de expected_range devem ser finitos.")
            if lower_float >= upper_float:
                raise ValueError("expected_range deve possuir mínimo menor que máximo.")
            return lower_float, upper_float
        if np.issubdtype(dtype, np.integer):
            limits = np.iinfo(dtype)
            return int(limits.min), int(limits.max)
        limits = np.finfo(dtype)
        return float(limits.min), float(limits.max)

    def print_image_info(self, image):
        """
        Prints metadata and statistical information about a given image.

        Parameters:
        - image (numpy.ndarray): The input image as a NumPy array.

        Functionality:
        1. Calls `self.get_image_info(image)`, which returns a dictionary containing:
        - height: Image height in pixels.
        - width: Image width in pixels.
        - dtype: Data type of the image (e.g., uint8, float32).
        - depth: Number of color channels (1 for grayscale, 3 for RGB, etc.).
        - min_value: Minimum pixel intensity in the image.
        - max_value: Maximum pixel intensity in the image.
        - mean: Mean pixel intensity.
        - std_dev: Standard deviation of pixel intensities.

        2. Iterates through the dictionary and prints each property in the format:
        `key: value`

        Example Output (for a 100x100 grayscale image):
        ```
        height: 100
        width: 100
        dtype: uint8
        depth: 1
        min_value: 12
        max_value: 255
        mean: 127.56
        std_dev: 32.78
        ```
        """
        info = self.get_image_info(image)
        # Print results
        for key, value in info.items():
            print(f"{key}: {value}")

    # Image Similarity - Normalized Correlation Coefficient
    def compute_image_similarity(self, image1, image2):
        """
        Computes the similarity between two images using normalized
        correlation coefficients.

        Parameters:
        - image1 (numpy.ndarray): First input image.
        - image2 (numpy.ndarray): Second input image.

        Returns:
        - float: Normalized correlation coefficient between the two images.
        """
        if image1.shape != image2.shape:
            raise ValueError(
                "Images must have the same dimensions for similarity computation."
            )

        # Convert to grayscale if necessary
        if len(image1.shape) == 3:
            image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        if len(image2.shape) == 3:
            image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

        # Flatten images into 1D arrays
        image1_flat = image1.flatten()
        image2_flat = image2.flatten()

        # Compute normalized correlation coefficient
        correlation, _ = pearsonr(image1_flat, image2_flat)

        return correlation

    # Basic Statistics
    def calculate_mean(self, data):
        """Calculates the mean of the data."""
        return np.mean(data)

    def calculate_std(self, data):
        """Calculates the standard deviation of the data."""
        return np.std(data)

    def calculate_variance(self, data):
        """Calculates the variance of the data."""
        return np.var(data)

    # Regression Metrics
    def calculate_r2(self, y_true, y_pred):
        """Calculates the R-squared value."""
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2

    def calculate_adjusted_r2(self, y_true, y_pred, num_predictors):
        """Calculates the Adjusted R-squared value."""
        n = len(y_true)
        r2 = self.calculate_r2(y_true, y_pred)
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - num_predictors - 1))
        return adj_r2

    # Correlation
    def calculate_correlation(self, data1, data2, method="pearson"):
        """
        Calculates the correlation between two datasets.
        Supports Pearson and Spearman methods.
        """
        if method == "pearson":
            corr, _ = pearsonr(data1, data2)
        elif method == "spearman":
            corr, _ = spearmanr(data1, data2)
        else:
            raise ValueError("Invalid method. Choose 'pearson' or 'spearman'.")
        return corr

    # Principal Component Analysis
    def perform_pca(self, data, n_components=2):
        """
        Performs PCA on the given dataset.
        Returns the transformed data and explained variance ratio.
        """
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(data)
        return transformed_data, pca.explained_variance_ratio_

    # Time Series Analysis
    def calculate_moving_average(self, data, window_size):
        """Calculates the moving average of a time series."""
        return pd.Series(data).rolling(window=window_size).mean().to_numpy()

    def calculate_exponential_moving_average(self, data, alpha):
        """Calculates the exponential moving average of a time series."""
        return pd.Series(data).ewm(alpha=alpha).mean().to_numpy()

    def calculate_autocorrelation(self, data, lag=1):
        """Calculates the autocorrelation of a time series for a given lag."""
        n = len(data)
        mean = np.mean(data)
        autocovariance = np.sum((data[: n - lag] - mean) * (data[lag:] - mean)) / n
        variance = np.var(data)
        return autocovariance / variance

    def benjamini_hochberg(p_values, fdr=0.05):
        """
        Perform Benjamini-Hochberg correction.

        Parameters:
        - p_values: List or numpy array of p-values.
        - fdr: Desired false discovery rate (default is 0.05).

        Returns:
        - A boolean array indicating which hypotheses are significant.
        """
        p_values = np.array(p_values)
        m = len(p_values)  # Total number of tests
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        thresholds = (np.arange(1, m + 1) / m) * fdr

        # Determine significant p-values
        significant = sorted_p_values <= thresholds
        if significant.any():
            max_significant_index = np.where(significant)[0].max()
            significant[: max_significant_index + 1] = True

        # Map significance back to the original order
        result = np.zeros_like(p_values, dtype=bool)
        result[sorted_indices] = significant
        return result
