"""Extração didática de bordas no domínio espacial."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Literal

import cv2 as cv
import numpy as np

from .image_preprocessor import ImagePreprocessor

SobelDirection = Literal["x", "y", "magnitude"]


class FeatureExtractor:
    """Calcula respostas de Sobel, Laplaciano e bordas de Canny.

    Sobel e Laplaciano aceitam grayscale 2D e retornam ``float64`` sem
    clipping. Canny aceita grayscale ``uint8`` e retorna uma máscara ``uint8``
    com valores 0 e 255. Imagens coloridas são rejeitadas explicitamente.
    """

    def __init__(self) -> None:
        self._spatial_filter = ImagePreprocessor()

    def sobel(
        self,
        image: np.ndarray,
        direction: SobelDirection = "magnitude",
    ) -> np.ndarray:
        """Calcula Sobel horizontal, vertical ou magnitude do gradiente.

        Kernels 3x3 são aplicados por correlação com padding zero.
        ``direction='x'`` mede variações nas colunas; ``'y'``, nas linhas;
        ``'magnitude'`` retorna ``hypot(Gx, Gy)``.
        """
        if not isinstance(direction, str):
            raise TypeError("direction deve ser uma string.")
        if direction not in {"x", "y", "magnitude"}:
            raise ValueError("direction deve ser 'x', 'y' ou 'magnitude'.")

        kernel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        kernel_y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        if direction == "x":
            return self._spatial_filter.correlate(image, kernel_x)
        if direction == "y":
            return self._spatial_filter.correlate(image, kernel_y)
        gradient_x = self._spatial_filter.correlate(image, kernel_x)
        gradient_y = self._spatial_filter.correlate(image, kernel_y)
        return np.hypot(gradient_x, gradient_y)

    def laplacian(self, image: np.ndarray) -> np.ndarray:
        """Calcula o Laplaciano 4-conectado com padding constante zero.

        A saída ``float64`` não sofre clipping e pode conter valores negativos.
        """
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        return self._spatial_filter.correlate(image, kernel)

    def canny(
        self,
        image: np.ndarray,
        low_threshold: Real,
        high_threshold: Real,
        *,
        aperture_size: int = 3,
        l2_gradient: bool = False,
    ) -> np.ndarray:
        """Detecta bordas com Canny usando limiares explícitos.

        Args:
            image: Imagem grayscale 2D com dtype ``uint8``.
            low_threshold: Limiar inferior finito e não negativo.
            high_threshold: Limiar superior finito e maior que o inferior.
            aperture_size: Tamanho 3, 5 ou 7 do Sobel interno.
            l2_gradient: Se verdadeiro, usa norma L2 para o gradiente.

        Returns:
            Máscara ``uint8`` de mesmo shape, com valores 0 e 255.

        O cálculo usa a extensão de borda interna do Canny do OpenCV. Uma
        imagem constante, inclusive em seus limites, não produz bordas.
        """
        self._spatial_filter._validate_image(image, allow_color=False)
        if image.dtype != np.dtype(np.uint8):
            raise TypeError("canny aceita somente imagens uint8.")
        low = self._validate_threshold(low_threshold, "low_threshold")
        high = self._validate_threshold(high_threshold, "high_threshold")
        if low >= high:
            raise ValueError("low_threshold deve ser menor que high_threshold.")
        if (
            isinstance(aperture_size, (bool, np.bool_))
            or not isinstance(aperture_size, Integral)
            or aperture_size not in {3, 5, 7}
        ):
            raise ValueError("aperture_size deve ser 3, 5 ou 7.")
        if not isinstance(l2_gradient, (bool, np.bool_)):
            raise TypeError("l2_gradient deve ser booleano.")
        return cv.Canny(
            image,
            threshold1=low,
            threshold2=high,
            apertureSize=int(aperture_size),
            L2gradient=bool(l2_gradient),
        )

    def extract_edges(
        self,
        image: np.ndarray,
        low_threshold: Real = 100,
        high_threshold: Real = 200,
        *,
        aperture_size: int = 3,
        l2_gradient: bool = False,
    ) -> np.ndarray:
        """Executa Canny preservando os limiares padrão da API legada."""
        return self.canny(
            image,
            low_threshold,
            high_threshold,
            aperture_size=aperture_size,
            l2_gradient=l2_gradient,
        )

    @staticmethod
    def _validate_threshold(value: Real, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} deve ser um número real.")
        result = float(value)
        if not np.isfinite(result) or result < 0:
            raise ValueError(f"{name} deve ser finito e não negativo.")
        return result
