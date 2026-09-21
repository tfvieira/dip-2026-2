"""Degradação controlada e restauração introdutória de imagens.

As operações aceitam imagens grayscale ``(H, W)`` ou coloridas ``(H, W, 3)``
em ``uint8``, ``uint16``, ``int16``, ``float32`` ou ``float64``. Elas preservam
shape, não alteram a entrada e retornam ``float64`` sem clipping implícito.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Literal

import numpy as np

from dip_toolkit.modules.image_preprocessor import ImagePreprocessor

Direction = Literal["horizontal", "vertical"]
_SUPPORTED_DTYPES = frozenset(
    np.dtype(dtype) for dtype in (np.uint8, np.uint16, np.int16, np.float32, np.float64)
)


class ImageRestorer:
    """Aplica degradações reproduzíveis e mede o erro de restauração.

    Métodos aleatórios aceitam ``seed`` ou ``rng``, mas nunca os dois. A mesma
    seed gera o mesmo resultado e o estado de um ``Generator`` é avançado.
    Filtros básicos continuam sendo fornecidos pela DIP-06.
    """

    def add_gaussian_noise(
        self,
        image: np.ndarray,
        mean: Real = 0.0,
        std: Real = 10.0,
        *,
        seed: int | np.integer | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Soma ruído gaussiano finito, com desvio positivo e sem clipping."""
        self._validate_image(image, "image")
        mean = self._validate_real(mean, "mean")
        std = self._validate_positive_real(std, "std")
        noise = self._resolve_rng(seed=seed, rng=rng).normal(mean, std, image.shape)
        return image.astype(np.float64, copy=True) + noise

    def add_salt_and_pepper_noise(
        self,
        image: np.ndarray,
        salt_prob: Real = 0.05,
        pepper_prob: Real = 0.05,
        *,
        salt_value: Real | None = None,
        pepper_value: Real | None = None,
        seed: int | np.integer | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Aplica impulsos exclusivos de sal e pimenta sem clipping.

        Para inteiros, os níveis padrão são mínimo e máximo do dtype; para
        floats, são ``0.0`` e ``1.0``. Em imagens coloridas, um pixel recebe a
        mesma classe em todos os canais.
        """
        self._validate_image(image, "image")
        salt_prob = self._validate_probability(salt_prob, "salt_prob")
        pepper_prob = self._validate_probability(pepper_prob, "pepper_prob")
        if salt_prob + pepper_prob > 1.0:
            raise ValueError("salt_prob + pepper_prob não pode ultrapassar 1.")
        default_pepper, default_salt = self._default_impulse_levels(image.dtype)
        salt = self._validate_real(
            default_salt if salt_value is None else salt_value,
            "salt_value",
        )
        pepper = self._validate_real(
            default_pepper if pepper_value is None else pepper_value,
            "pepper_value",
        )
        result = image.astype(np.float64, copy=True)
        values = self._resolve_rng(seed=seed, rng=rng).random(image.shape[:2])
        pepper_mask = values < pepper_prob
        salt_mask = (values >= pepper_prob) & (values < pepper_prob + salt_prob)
        result[pepper_mask] = pepper
        result[salt_mask] = salt
        return result

    def motion_blur_kernel(
        self,
        kernel_size: int = 3,
        direction: Direction = "horizontal",
    ) -> np.ndarray:
        """Cria kernel normalizado de motion blur horizontal ou vertical."""
        size = self._validate_kernel_size(kernel_size)
        self._validate_direction(direction)
        kernel = np.zeros((size, size), dtype=np.float64)
        if direction == "horizontal":
            kernel[size // 2, :] = 1.0 / size
        else:
            kernel[:, size // 2] = 1.0 / size
        return kernel

    def apply_motion_blur(
        self,
        image: np.ndarray,
        kernel_size: int = 3,
        direction: Direction = "horizontal",
    ) -> np.ndarray:
        """Aplica motion blur com a correlação espacial da DIP-06."""
        self._validate_image(image, "image")
        kernel = self.motion_blur_kernel(kernel_size, direction)
        preprocessor = ImagePreprocessor()
        if image.ndim == 2:
            return preprocessor.correlate(image, kernel)
        return np.stack(
            [
                preprocessor.correlate(image[..., channel], kernel)
                for channel in range(3)
            ],
            axis=-1,
        )

    def mse(self, reference: np.ndarray, result: np.ndarray) -> float:
        """Calcula ``mean((reference - result) ** 2)`` em ``float64``."""
        self._validate_image(reference, "reference")
        self._validate_image(result, "result")
        if reference.shape != result.shape:
            raise ValueError("reference e result devem ter o mesmo shape.")
        difference = reference.astype(np.float64) - result.astype(np.float64)
        return float(np.mean(difference**2))

    @staticmethod
    def _validate_image(image: np.ndarray, name: str) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} deve ser um array NumPy.")
        if image.ndim not in {2, 3}:
            raise ValueError(f"{name} deve ter duas ou três dimensões.")
        if any(dimension <= 0 for dimension in image.shape):
            raise ValueError(f"{name} não pode estar vazio.")
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError(f"{name} colorida deve possuir exatamente três canais.")
        if image.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} usa dtype não suportado: {image.dtype}.")
        if not np.all(np.isfinite(image)):
            raise ValueError(f"{name} deve conter somente valores finitos.")

    @staticmethod
    def _validate_real(value: Real, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} deve ser um número real.")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} deve ser finito.")
        return result

    @classmethod
    def _validate_positive_real(cls, value: Real, name: str) -> float:
        result = cls._validate_real(value, name)
        if result <= 0:
            raise ValueError(f"{name} deve ser maior que zero.")
        return result

    @classmethod
    def _validate_probability(cls, value: Real, name: str) -> float:
        result = cls._validate_real(value, name)
        if not 0.0 <= result <= 1.0:
            raise ValueError(f"{name} deve estar no intervalo [0, 1].")
        return result

    @staticmethod
    def _validate_kernel_size(kernel_size: int) -> int:
        if isinstance(kernel_size, (bool, np.bool_)) or not isinstance(
            kernel_size, Integral
        ):
            raise TypeError("kernel_size deve ser um número inteiro.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size deve ser positivo e ímpar.")
        return int(kernel_size)

    @staticmethod
    def _validate_direction(direction: Direction) -> None:
        if not isinstance(direction, str):
            raise TypeError("direction deve ser uma string.")
        if direction not in {"horizontal", "vertical"}:
            raise ValueError("direction deve ser 'horizontal' ou 'vertical'.")

    @staticmethod
    def _default_impulse_levels(dtype: np.dtype) -> tuple[float, float]:
        if np.issubdtype(dtype, np.integer):
            limits = np.iinfo(dtype)
            return float(limits.min), float(limits.max)
        return 0.0, 1.0

    @staticmethod
    def _resolve_rng(
        *,
        seed: int | np.integer | None,
        rng: np.random.Generator | None,
    ) -> np.random.Generator:
        if seed is not None and rng is not None:
            raise ValueError("Use seed ou rng, não ambos.")
        if rng is not None:
            if not isinstance(rng, np.random.Generator):
                raise TypeError("rng deve ser uma instância de numpy.random.Generator.")
            return rng
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
                raise TypeError("seed deve ser um número inteiro não negativo.")
            if seed < 0:
                raise ValueError("seed deve ser um número inteiro não negativo.")
            seed = int(seed)
        return np.random.default_rng(seed)
