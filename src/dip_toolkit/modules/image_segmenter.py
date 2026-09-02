"""Segmentação didática de imagens grayscale e regiões conectadas."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Literal, NamedTuple

import cv2 as cv
import numpy as np

from .feature_extractor import FeatureExtractor
from .morphology import BorderPolicy, MorphologyProcessor

Polarity = Literal["bright", "dark"]
RefinementOperation = Literal["opening", "closing"]


class OtsuSegmentation(NamedTuple):
    """Limiar determinado por Otsu e a máscara correspondente."""

    threshold: int
    mask: np.ndarray


class ConnectedComponents(NamedTuple):
    """Máscara, mapa de rótulos e identificadores de regiões.

    O fundo recebe o rótulo 0. Regiões recebem IDs consecutivos a partir de 1,
    seguindo a varredura linha a linha usada pelo OpenCV.
    """

    mask: np.ndarray
    labels: np.ndarray
    region_ids: tuple[int, ...]


class ImageSegmenter:
    """Segmenta imagens grayscale 2D ``uint8``.

    As máscaras usam 0 para fundo e 255 para primeiro plano. Imagens coloridas
    são rejeitadas explicitamente; converta-as para grayscale antes do uso.
    """

    def __init__(self) -> None:
        self._features = FeatureExtractor()
        self._morphology = MorphologyProcessor()

    def global_threshold(
        self, image: np.ndarray, threshold: int, *, polarity: Polarity = "bright"
    ) -> np.ndarray:
        """Aplica limiar global; a igualdade pertence ao primeiro plano."""
        image = self._validate_image(image)
        threshold = self._validate_intensity(threshold, "threshold")
        polarity = self._validate_polarity(polarity)
        comparison = image >= threshold if polarity == "bright" else image <= threshold
        return np.where(comparison, 255, 0).astype(np.uint8)

    def otsu(
        self, image: np.ndarray, *, polarity: Polarity = "bright"
    ) -> OtsuSegmentation:
        """Determina automaticamente um limiar de Otsu e o aplica."""
        image = self._validate_image(image)
        polarity = self._validate_polarity(polarity)
        threshold_type = (
            cv.THRESH_BINARY if polarity == "bright" else cv.THRESH_BINARY_INV
        )
        threshold, mask = cv.threshold(image, 0, 255, threshold_type | cv.THRESH_OTSU)
        return OtsuSegmentation(int(threshold), mask)

    def adaptive_threshold(
        self,
        image: np.ndarray,
        block_size: int,
        constant: Real = 0,
        *,
        polarity: Polarity = "bright",
    ) -> np.ndarray:
        """Segmenta pela média local de uma vizinhança ímpar menos constante."""
        image = self._validate_image(image)
        block_size = self._validate_block_size(block_size)
        constant = self._validate_constant(constant)
        polarity = self._validate_polarity(polarity)
        threshold_type = (
            cv.THRESH_BINARY if polarity == "bright" else cv.THRESH_BINARY_INV
        )
        return cv.adaptiveThreshold(
            image,
            255,
            cv.ADAPTIVE_THRESH_MEAN_C,
            threshold_type,
            block_size,
            constant,
        )

    def edge_segmentation(
        self,
        image: np.ndarray,
        low_threshold: Real,
        high_threshold: Real,
        *,
        aperture_size: int = 3,
        l2_gradient: bool = False,
    ) -> np.ndarray:
        """Segmenta bordas reutilizando o Canny parametrizável da DIP-06."""
        self._validate_image(image)
        return self._features.canny(
            image,
            low_threshold,
            high_threshold,
            aperture_size=aperture_size,
            l2_gradient=l2_gradient,
        )

    def refine_mask(
        self,
        mask: np.ndarray,
        element: np.ndarray,
        operations: Sequence[RefinementOperation],
        *,
        iterations: int = 1,
        border: BorderPolicy = "constant",
    ) -> np.ndarray:
        """Compõe abertura e fechamento já implementados na DIP-09."""
        self._morphology._validate_mask(mask)
        self._morphology._validate_element(element)
        if isinstance(operations, str) or not isinstance(operations, Sequence):
            raise TypeError("operations deve ser uma sequência de operações.")
        if not operations:
            raise ValueError("operations deve conter ao menos uma operação.")
        result = mask.copy()
        for operation in operations:
            if operation == "opening":
                result = self._morphology.opening(
                    result, element, iterations=iterations, border=border
                )
            elif operation == "closing":
                result = self._morphology.closing(
                    result, element, iterations=iterations, border=border
                )
            else:
                raise ValueError("operations aceita somente 'opening' e 'closing'.")
        return result

    def connected_components(
        self, mask: np.ndarray, *, connectivity: Literal[4, 8] = 8
    ) -> ConnectedComponents:
        """Rotula regiões de primeiro plano com conectividade 4 ou 8."""
        mask = self._morphology._validate_mask(mask)
        if (
            isinstance(connectivity, (bool, np.bool_))
            or not isinstance(connectivity, Integral)
            or connectivity not in {4, 8}
        ):
            raise ValueError("connectivity deve ser 4 ou 8.")
        total, labels = cv.connectedComponents(
            mask, connectivity=int(connectivity), ltype=cv.CV_32S
        )
        return ConnectedComponents(mask.copy(), labels, tuple(range(1, total)))

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("image deve ser um array NumPy.")
        if image.ndim != 2 or image.size == 0:
            raise ValueError("image deve ser uma imagem grayscale 2D não vazia.")
        if image.dtype != np.dtype(np.uint8):
            raise TypeError("image deve possuir dtype uint8.")
        return image

    @staticmethod
    def _validate_intensity(value: int, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} deve ser um número inteiro entre 0 e 255.")
        if not 0 <= value <= 255:
            raise ValueError(f"{name} deve estar entre 0 e 255.")
        return int(value)

    @staticmethod
    def _validate_polarity(polarity: Polarity) -> Polarity:
        if not isinstance(polarity, str) or polarity not in {"bright", "dark"}:
            raise ValueError("polarity deve ser 'bright' ou 'dark'.")
        return polarity

    @staticmethod
    def _validate_block_size(block_size: int) -> int:
        if isinstance(block_size, (bool, np.bool_)) or not isinstance(
            block_size, Integral
        ):
            raise TypeError("block_size deve ser um número inteiro ímpar.")
        if block_size <= 1 or block_size % 2 == 0:
            raise ValueError("block_size deve ser ímpar e maior que 1.")
        return int(block_size)

    @staticmethod
    def _validate_constant(constant: Real) -> float:
        if isinstance(constant, (bool, np.bool_)) or not isinstance(constant, Real):
            raise TypeError("constant deve ser um número real.")
        constant = float(constant)
        if not np.isfinite(constant):
            raise ValueError("constant deve ser finito.")
        return constant
