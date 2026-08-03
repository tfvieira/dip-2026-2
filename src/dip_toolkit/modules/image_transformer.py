from __future__ import annotations

from numbers import Integral, Real
from typing import Literal

import cv2 as cv
import numpy as np

InterpolationMethod = Literal["nearest", "bilinear", "bicubic"]
ImageShape = tuple[int, int] | tuple[int, int, int]


class ImageTransformer:
    """Aplica transformações geométricas e de representação em imagens."""

    def __init__(self):
        pass

    def translate(self, image, tx, ty):
        """Translates the image by tx (x-axis) and ty (y-axis)."""
        rows, cols = image.shape[:2]
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv.warpAffine(image, translation_matrix, (cols, rows))

    def rotate(self, image, angle, center=None, scale=1.0):
        """Rotates the image around a center point."""
        rows, cols = image.shape[:2]
        if center is None:
            center = (cols // 2, rows // 2)
        rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
        return cv.warpAffine(image, rotation_matrix, (cols, rows))

    def mirror(self, image, axis="x"):
        """Mirrors the image along the specified axis (x, y, or both)."""
        if axis == "x":
            return cv.flip(image, 0)  # Flip vertically
        elif axis == "y":
            return cv.flip(image, 1)  # Flip horizontally
        elif axis == "xy":
            return cv.flip(image, -1)  # Flip both vertically and horizontally
        else:
            raise ValueError("Invalid axis. Choose 'x', 'y', or 'xy'.")

    def stretch(self, image, fx, fy):
        """Stretches (scales) the image along x and y axes."""
        return cv.resize(image, None, fx=fx, fy=fy, interpolation=cv.INTER_LINEAR)

    def crop(self, image, x_start, y_start, width, height):
        """Crops the image to the specified rectangle."""
        return image[y_start : y_start + height, x_start : x_start + width]

    def resize(self, image, width, height):
        """Resizes the image to the specified dimensions."""
        return cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)

    def resample(
        self,
        image: np.ndarray,
        output_shape: tuple[int, int],
        interpolation: InterpolationMethod = "bilinear",
    ) -> np.ndarray:
        """Reamostra uma imagem com interpolação nearest, bilinear ou bicúbica.

        ``output_shape`` é informado como ``(altura, largura)``. A operação
        retorna um novo array, preservando o dtype e a quantidade de canais da
        imagem de entrada. Os dtypes suportados pelo OpenCV neste método são
        ``uint8``, ``uint16``, ``int16``, ``float32`` e ``float64``.

        Args:
            image: Imagem 2D em escala de cinza ou 3D com três canais.
            output_shape: Altura e largura positivas desejadas.
            interpolation: Método ``"nearest"``, ``"bilinear"`` ou
                ``"bicubic"``.

        Raises:
            TypeError: Se a imagem, dimensões ou dtype não forem suportados.
            ValueError: Se o shape ou o interpolador forem inválidos.
        """

        self._validate_image(image)
        self._validate_resampling_dtype(image.dtype)
        height, width = self._validate_output_shape(output_shape)
        interpolation_flag = self._resolve_interpolation(interpolation)
        return cv.resize(
            image,
            (width, height),
            interpolation=interpolation_flag,
        )

    def quantize_uniform(
        self,
        image: np.ndarray,
        levels: int,
        *,
        value_range: tuple[Real, Real] | None = None,
    ) -> np.ndarray:
        """Quantiza uniformemente uma imagem em níveis igualmente espaçados.

        A imagem precisa estar na faixa ``value_range``. Quando ela não é
        informada, usa-se o intervalo completo do dtype inteiro ou ``[0.0, 1.0]``
        para dtypes de ponto flutuante. Os valores de saída pertencem a ``levels``
        níveis, incluindo as duas extremidades da faixa, e o dtype é preservado.
        A entrada não é alterada.

        Args:
            image: Imagem 2D em escala de cinza ou 3D com três canais.
            levels: Número inteiro de níveis, maior ou igual a dois.
            value_range: Faixa inclusiva ``(mínimo, máximo)`` da imagem.

        Raises:
            TypeError: Se a imagem, os níveis ou a faixa tiverem tipos inválidos.
            ValueError: Se os níveis, a faixa ou os valores da imagem forem
                incompatíveis.
        """

        self._validate_image(image)
        validated_levels = self._validate_levels(levels)
        lower, upper = self._validate_value_range(value_range, image.dtype)

        if np.any(image < lower) or np.any(image > upper):
            raise ValueError("image contém valores fora de value_range.")

        normalized = (image.astype(np.float64) - lower) / (upper - lower)
        indices = np.floor(normalized * validated_levels).astype(np.int64)
        indices = np.clip(indices, 0, validated_levels - 1)
        quantized = lower + indices * (upper - lower) / (validated_levels - 1)

        if np.issubdtype(image.dtype, np.integer):
            quantized = np.rint(quantized)
        return quantized.astype(image.dtype, copy=False)

    @staticmethod
    def _validate_image(image: np.ndarray) -> ImageShape:
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
        return image.shape  # type: ignore[return-value]

    @staticmethod
    def _validate_output_shape(output_shape: tuple[int, int]) -> tuple[int, int]:
        if not isinstance(output_shape, tuple) or len(output_shape) != 2:
            raise TypeError("output_shape deve ser uma tupla (altura, largura).")
        if any(
            isinstance(dimension, (bool, np.bool_))
            or not isinstance(dimension, Integral)
            for dimension in output_shape
        ):
            raise TypeError("As dimensões de output_shape devem ser inteiros.")
        if any(dimension <= 0 for dimension in output_shape):
            raise ValueError("As dimensões de output_shape devem ser positivas.")
        return int(output_shape[0]), int(output_shape[1])

    @staticmethod
    def _validate_resampling_dtype(dtype: np.dtype) -> None:
        supported_dtypes = {
            np.dtype(np.uint8),
            np.dtype(np.uint16),
            np.dtype(np.int16),
            np.dtype(np.float32),
            np.dtype(np.float64),
        }
        if dtype not in supported_dtypes:
            raise TypeError(
                "resample aceita somente uint8, uint16, int16, float32 ou float64."
            )

    @staticmethod
    def _resolve_interpolation(interpolation: InterpolationMethod) -> int:
        methods = {
            "nearest": cv.INTER_NEAREST,
            "bilinear": cv.INTER_LINEAR,
            "bicubic": cv.INTER_CUBIC,
        }
        if not isinstance(interpolation, str):
            raise TypeError("interpolation deve ser uma string.")
        try:
            return methods[interpolation]
        except KeyError as error:
            raise ValueError(
                "interpolation deve ser 'nearest', 'bilinear' ou 'bicubic'."
            ) from error

    @staticmethod
    def _validate_levels(levels: int) -> int:
        if isinstance(levels, (bool, np.bool_)) or not isinstance(levels, Integral):
            raise TypeError("levels deve ser um número inteiro.")
        if levels < 2:
            raise ValueError("levels deve ser maior ou igual a dois.")
        return int(levels)

    @staticmethod
    def _validate_value_range(
        value_range: tuple[Real, Real] | None,
        dtype: np.dtype,
    ) -> tuple[float, float]:
        if value_range is None:
            if np.issubdtype(dtype, np.integer):
                limits = np.iinfo(dtype)
                return float(limits.min), float(limits.max)
            return 0.0, 1.0
        if not isinstance(value_range, tuple) or len(value_range) != 2:
            raise TypeError("value_range deve ser uma tupla (mínimo, máximo).")
        lower, upper = value_range
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in value_range
        ):
            raise TypeError("Os limites de value_range devem ser números reais.")
        lower_float, upper_float = float(lower), float(upper)
        if not np.isfinite(lower_float) or not np.isfinite(upper_float):
            raise ValueError("Os limites de value_range devem ser finitos.")
        if lower_float >= upper_float:
            raise ValueError("value_range deve possuir mínimo menor que máximo.")
        return lower_float, upper_float
