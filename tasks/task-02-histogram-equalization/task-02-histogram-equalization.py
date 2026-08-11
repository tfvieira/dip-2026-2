"""Task 02: equalizar uma imagem grayscale uint8 com NumPy."""

from __future__ import annotations

import numpy as np


def compute_histogram(image: np.ndarray) -> np.ndarray:
    """Retorna as contagens de intensidades de uma imagem grayscale uint8."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image deve ser um array NumPy.")
    if image.ndim != 2:
        raise ValueError("image deve ser uma imagem grayscale 2D.")
    if image.dtype != np.uint8:
        raise TypeError("image deve possuir dtype uint8.")
    return np.bincount(image.ravel(), minlength=256)


def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    """Retorna a CDF acumulada de um histograma."""
    return np.cumsum(histogram, dtype=np.int64)


def equalize_grayscale(image: np.ndarray) -> np.ndarray:
    """Equaliza uma imagem grayscale uint8 a partir de histograma e CDF.

    Args:
        image: Array NumPy 2D com dtype ``uint8``.

    Returns:
        Nova imagem equalizada, com o mesmo shape e dtype da entrada.
    """
    # ### START CODE HERE ###
    # 1. Valide image usando compute_histogram.
    # 2. Preserve imagens constantes retornando uma cópia.
    # 3. Calcule histograma e CDF.
    # 4. Encontre cdf_min, o primeiro valor positivo da CDF.
    # 5. Construa uma LUT uint8 com a fórmula:
    #    round((cdf - cdf_min) / (image.size - cdf_min) * 255)
    # 6. Aplique a LUT à imagem e retorne o resultado.
    raise NotImplementedError("Implemente equalize_grayscale.")
    # ### END CODE HERE ###


def run_tests() -> None:
    """Executa os testes automáticos da atividade."""
    image = np.array(
        [
            [0, 0, 1, 1],
            [2, 2, 3, 3],
        ],
        dtype=np.uint8,
    )
    original = image.copy()

    equalized = equalize_grayscale(image)
    expected = np.array(
        [
            [0, 0, 85, 85],
            [170, 170, 255, 255],
        ],
        dtype=np.uint8,
    )

    assert equalized.shape == image.shape
    assert equalized.dtype == np.uint8
    assert np.array_equal(equalized, expected)
    assert np.array_equal(image, original)
    assert equalized is not image

    constant = np.full((3, 4), 73, dtype=np.uint8)
    constant_equalized = equalize_grayscale(constant)
    assert np.array_equal(constant_equalized, constant)
    assert constant_equalized is not constant

    print("Test passed!")


if __name__ == "__main__":
    run_tests()
