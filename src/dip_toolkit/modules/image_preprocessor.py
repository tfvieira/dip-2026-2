"""Filtragem espacial didática baseada em correlação 2D."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Literal

import cv2 as cv
import numpy as np

FilterType = Literal["blur", "median"]
_SUPPORTED_DTYPES = frozenset(
    np.dtype(dtype) for dtype in (np.uint8, np.uint16, np.int16, np.float32, np.float64)
)


class ImagePreprocessor:
    """Aplica filtros espaciais a imagens NumPy.

    ``correlate`` usa o kernel sem inversão e padding constante zero. Uma
    convolução exige inverter o kernel nos dois eixos antes da aplicação.
    Operações lineares retornam ``float64`` sem clipping; a mediana preserva o
    dtype. Filtros de alto nível processam três canais separadamente.
    """

    def resize_image(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Redimensiona uma imagem para ``(height, width)``.

        Método legado preservado por compatibilidade. Em novos fluxos, prefira
        ``ImageTransformer.resample``.
        """
        self._validate_image(image)
        width = self._validate_positive_integer(width, "width", odd=False)
        height = self._validate_positive_integer(height, "height", odd=False)
        return cv.resize(image, (width, height))

    def apply_filter(
        self,
        image: np.ndarray,
        filter_type: FilterType = "blur",
        **kwargs: int,
    ) -> np.ndarray:
        """Aplica Gauss ou mediana preservando o comportamento legado.

        ``blur`` delega diretamente a ``cv.GaussianBlur`` com sigma automático.
        Para ``median``, um tamanho par é promovido ao próximo ímpar antes da
        chamada a ``cv.medianBlur``. Ambas as operações preservam o dtype.
        """
        ksize = kwargs.get("ksize", 5)

        if filter_type == "blur":
            return cv.GaussianBlur(image, (ksize, ksize), 0)
        if filter_type == "median":
            if ksize % 2 == 0:
                ksize += 1
            return cv.medianBlur(image, ksize)
        raise ValueError("filter_type deve ser 'blur' ou 'median'.")

    def correlate(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Calcula correlação 2D grayscale com padding constante zero.

        Para cada pixel, multiplica a vizinhança pelo kernel sem invertê-lo e
        soma os produtos. A saída mantém o shape, usa ``float64`` e não sofre
        normalização nem clipping.

        Args:
            image: Imagem grayscale 2D, não vazia, numérica e finita.
            kernel: Array NumPy 2D, não vazio, numérico, finito e com dimensões
                ímpares.

        Returns:
            Novo array ``float64`` com o mesmo shape de ``image``.

        Raises:
            TypeError: Se imagem ou kernel tiver tipo ou dtype inválido.
            ValueError: Se shape, conteúdo ou dimensões forem inválidos.
        """
        self._validate_image(image, allow_color=False)
        return self._correlate_2d(image, self._validate_kernel(kernel))

    def mean_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Suaviza com kernel de média e extensão da borda por pixel próximo.

        O kernel gerado tem soma um, preserva imagens constantes e produz uma
        saída ``float64`` sem clipping.
        """
        self._validate_image(image)
        size = self._validate_positive_integer(kernel_size, "kernel_size", odd=True)
        kernel = np.ones((size, size), dtype=np.float64) / (size * size)
        return self._filter_channels(image, kernel, padding="edge")

    def gaussian_filter(
        self,
        image: np.ndarray,
        kernel_size: int = 3,
        sigma: Real = 1.0,
    ) -> np.ndarray:
        """Suaviza com kernel Gaussiano configurável e borda estendida.

        O tamanho deve ser positivo e ímpar; ``sigma`` deve ser finito e
        positivo. O kernel tem soma um e a saída é ``float64`` sem clipping.
        """
        self._validate_image(image)
        size = self._validate_positive_integer(kernel_size, "kernel_size", odd=True)
        sigma = self._validate_positive_real(sigma, "sigma")
        coordinates = np.arange(size, dtype=np.float64) - size // 2
        yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return self._filter_channels(image, kernel, padding="edge")

    def median_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Aplica mediana local com borda estendida, preservando dtype e shape."""
        self._validate_image(image)
        size = self._validate_positive_integer(kernel_size, "kernel_size", odd=True)
        if image.ndim == 2:
            return self._median_2d(image, size)
        channels = [self._median_2d(image[..., index], size) for index in range(3)]
        return np.stack(channels, axis=-1)

    def sharpen(self, image: np.ndarray, amount: Real = 1.0) -> np.ndarray:
        """Realça detalhes com ganho positivo e sem overflow silencioso.

        O kernel é ``[[0,-a,0],[-a,1+4a,-a],[0,-a,0]]``. A saída ``float64``
        não sofre clipping e a borda repete o pixel mais próximo.
        """
        self._validate_image(image)
        amount = self._validate_positive_real(amount, "amount")
        kernel = np.array(
            [
                [0.0, -amount, 0.0],
                [-amount, 1.0 + 4.0 * amount, -amount],
                [0.0, -amount, 0.0],
            ]
        )
        return self._filter_channels(image, kernel, padding="edge")

    @classmethod
    def _filter_channels(
        cls,
        image: np.ndarray,
        kernel: np.ndarray,
        *,
        padding: Literal["constant", "edge"],
    ) -> np.ndarray:
        if image.ndim == 2:
            return cls._correlate_2d(image, kernel, padding=padding)
        channels = [
            cls._correlate_2d(image[..., index], kernel, padding=padding)
            for index in range(3)
        ]
        return np.stack(channels, axis=-1)

    @staticmethod
    def _correlate_2d(
        image: np.ndarray,
        kernel: np.ndarray,
        *,
        padding: Literal["constant", "edge"] = "constant",
    ) -> np.ndarray:
        image_float = image.astype(np.float64, copy=False)
        kernel_float = kernel.astype(np.float64, copy=False)
        row_radius = kernel.shape[0] // 2
        column_radius = kernel.shape[1] // 2
        pad_width = ((row_radius, row_radius), (column_radius, column_radius))
        if padding == "constant":
            padded = np.pad(image_float, pad_width, mode="constant")
        else:
            padded = np.pad(image_float, pad_width, mode="edge")

        result = np.empty(image.shape, dtype=np.float64)
        kernel_height, kernel_width = kernel.shape
        for row in range(image.shape[0]):
            for column in range(image.shape[1]):
                neighborhood = padded[
                    row : row + kernel_height,
                    column : column + kernel_width,
                ]
                result[row, column] = np.sum(neighborhood * kernel_float)
        return result

    @staticmethod
    def _median_2d(image: np.ndarray, kernel_size: int) -> np.ndarray:
        radius = kernel_size // 2
        padded = np.pad(image, radius, mode="edge")
        result = np.empty_like(image)
        for row in range(image.shape[0]):
            for column in range(image.shape[1]):
                neighborhood = padded[
                    row : row + kernel_size,
                    column : column + kernel_size,
                ]
                result[row, column] = np.median(neighborhood)
        return result

    @staticmethod
    def _validate_image(image: np.ndarray, *, allow_color: bool = True) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image deve ser um array NumPy.")
        allowed_dimensions = {2, 3} if allow_color else {2}
        if image.ndim not in allowed_dimensions:
            requirement = (
                "duas dimensões" if not allow_color else "duas ou três dimensões"
            )
            raise ValueError(f"image deve ter {requirement}.")
        if any(dimension <= 0 for dimension in image.shape):
            raise ValueError("image não pode possuir dimensões vazias.")
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError("Imagens coloridas devem possuir exatamente três canais.")
        if image.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"image usa dtype não suportado: {image.dtype}. "
                "Dtypes suportados: uint8, uint16, int16, float32 e float64."
            )
        if not np.all(np.isfinite(image)):
            raise ValueError("image deve conter somente valores finitos.")

    @staticmethod
    def _validate_kernel(kernel: np.ndarray) -> np.ndarray:
        if not isinstance(kernel, np.ndarray):
            raise TypeError("kernel deve ser um array NumPy.")
        if kernel.ndim != 2:
            raise ValueError("kernel deve ter duas dimensões.")
        if kernel.size == 0:
            raise ValueError("kernel não pode estar vazio.")
        if np.issubdtype(kernel.dtype, np.bool_) or not np.issubdtype(
            kernel.dtype, np.number
        ):
            raise TypeError("kernel deve conter números reais.")
        if np.iscomplexobj(kernel):
            raise TypeError("kernel deve conter números reais.")
        if not np.all(np.isfinite(kernel)):
            raise ValueError("kernel deve conter somente valores finitos.")
        if any(dimension % 2 == 0 for dimension in kernel.shape):
            raise ValueError("As dimensões do kernel devem ser ímpares.")
        return kernel

    @staticmethod
    def _validate_positive_integer(value: int, name: str, *, odd: bool) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} deve ser um número inteiro.")
        if value <= 0:
            raise ValueError(f"{name} deve ser maior que zero.")
        if odd and value % 2 == 0:
            raise ValueError(f"{name} deve ser ímpar.")
        return int(value)

    @staticmethod
    def _validate_positive_real(value: Real, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} deve ser um número real.")
        result = float(value)
        if not np.isfinite(result) or result <= 0:
            raise ValueError(f"{name} deve ser finito e maior que zero.")
        return result

    @staticmethod
    def _validate_non_negative_real(value: Real, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} deve ser um número real.")
        result = float(value)
        if not np.isfinite(result) or result < 0:
            raise ValueError(f"{name} deve ser finito e maior ou igual a zero.")
        return result

    @staticmethod
    def _restore_dtype(result: np.ndarray, dtype: np.dtype) -> np.ndarray:
        if np.issubdtype(dtype, np.integer):
            limits = np.iinfo(dtype)
            result = np.clip(np.rint(result), limits.min, limits.max)
        return result.astype(dtype)
