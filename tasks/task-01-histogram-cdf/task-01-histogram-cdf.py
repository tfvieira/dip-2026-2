"""Task 01: calcular o histograma e a CDF normalizada de uma imagem grayscale."""

from __future__ import annotations

import numpy as np


def compute_histogram(image: np.ndarray) -> np.ndarray:
    """Retorna as contagens das 256 intensidades de uma imagem grayscale uint8.

    Args:
        image: Array NumPy 2D com dtype ``uint8``.

    Returns:
        Array de shape ``(256,)``. A posição ``i`` contém a quantidade de
        pixels com intensidade ``i``.
    """
    # ### START CODE HERE ###
    # Valide que a entrada e uma imagem grayscale uint8 e calcule as contagens
    # de cada intensidade de 0 a 255 usando NumPy.
    raise NotImplementedError("Implemente compute_histogram.")
    # ### END CODE HERE ###


def compute_normalized_cdf(histogram: np.ndarray) -> np.ndarray:
    """Retorna a função de distribuição acumulada normalizada.

    Args:
        histogram: Array unidimensional de contagens nao negativas.

    Returns:
        Array float com a mesma dimensão de ``histogram``. Para um histograma
        com pixels, o último valor e ``1.0``; para um histograma vazio, todos
        os valores sao ``0.0``.
    """
    # ### START CODE HERE ###
    # Calcule a soma acumulada e normalize-a pelo número total de pixels.
    # Se o total for zero, retorne um array de zeros com o mesmo shape.
    raise NotImplementedError("Implemente compute_normalized_cdf.")
    # ### END CODE HERE ###


def run_tests() -> None:
    """Executa os testes automáticos da atividade."""
    image = np.array(
        [
            [0, 0, 1, 3],
            [1, 2, 3, 3],
        ],
        dtype=np.uint8,
    )

    histogram = compute_histogram(image)
    expected_counts = np.zeros(256, dtype=np.int64)
    expected_counts[:4] = [2, 2, 1, 3]

    assert histogram.shape == (256,)
    assert np.array_equal(histogram, expected_counts)
    assert histogram.sum() == image.size

    cdf = compute_normalized_cdf(histogram)
    assert cdf.shape == histogram.shape
    assert np.allclose(cdf[:4], [0.25, 0.5, 0.625, 1.0])
    assert np.all(np.diff(cdf) >= 0.0)
    assert cdf[-1] == 1.0

    empty_cdf = compute_normalized_cdf(np.zeros(256, dtype=np.int64))
    assert np.array_equal(empty_cdf, np.zeros(256, dtype=float))

    print("Test passed!")


if __name__ == "__main__":
    run_tests()
