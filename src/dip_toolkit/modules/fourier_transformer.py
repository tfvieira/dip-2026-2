from __future__ import annotations

from numbers import Integral, Real

import numpy as np


class FourierTransformer:
    """Transformada de Fourier introdutória para imagens grayscale 2D.

    A API usa espectros complexos 2D centralizados e máscaras ideais binárias
    para apresentar os fundamentos de filtragem no domínio da frequência.
    """

    def __init__(self):
        pass

    def dft(self, image: np.ndarray) -> np.ndarray:
        """Calcula a DFT 2D e retorna um espectro complexo centralizado.

        A entrada deve ser uma imagem grayscale NumPy 2D, real, numérica, não
        vazia e finita. O resultado é um novo array ``complex128`` com o mesmo
        shape. A frequência zero (DC) fica no centro, conforme ``fftshift``.

        Args:
            image: Imagem grayscale 2D válida.

        Returns:
            Espectro complexo 2D centralizado.

        Raises:
            TypeError: Se a entrada ou o dtype não forem suportados.
            ValueError: Se shape, tamanho ou valores forem inválidos.
        """
        self._validate_image(image)
        image_float = image.astype(np.float64, copy=False)
        return np.fft.fftshift(np.fft.fft2(image_float))

    def idft(self, spectrum: np.ndarray) -> np.ndarray:
        """Reconstrói uma imagem real a partir de um espectro centralizado.

        Aplica ``ifftshift`` antes da transformada inversa e retorna somente a
        parte real em ``float64``, sem clipping ou conversão para ``uint8``.

        Args:
            spectrum: Espectro NumPy complexo 2D, centralizado e finito.

        Returns:
            Novo array real ``float64`` com o mesmo shape.

        Raises:
            TypeError: Se a entrada não for um espectro NumPy complexo.
            ValueError: Se shape, tamanho ou valores forem inválidos.
        """
        self._validate_spectrum(spectrum)
        spectrum_complex = spectrum.astype(np.complex128, copy=False)
        reconstructed = np.fft.ifft2(np.fft.ifftshift(spectrum_complex))
        return reconstructed.real.astype(np.float64, copy=False)

    def magnitude(self, spectrum: np.ndarray) -> np.ndarray:
        """Retorna a magnitude numérica ``float64`` de um espectro 2D.

        O resultado é não negativo e tem o mesmo shape. Nenhum logaritmo é
        aplicado; para exibição, use ``numpy.log1p`` explicitamente.
        """
        self._validate_spectrum(spectrum)
        return np.abs(spectrum).astype(np.float64, copy=False)

    def phase(self, spectrum: np.ndarray) -> np.ndarray:
        """Retorna a fase ``float64``, em radianos, de um espectro 2D."""
        self._validate_spectrum(spectrum)
        return np.angle(spectrum).astype(np.float64, copy=False)

    def ideal_low_pass_mask(
        self,
        shape: tuple[int, int],
        cutoff: float,
    ) -> np.ndarray:
        """Cria uma máscara passa-baixa ideal circular centralizada.

        O centro é ``(altura // 2, largura // 2)``, conforme ``fftshift``.
        Distâncias menores ou iguais ao cutoff recebem 1. O cutoff, em bins,
        deve estar entre zero e a distância do centro ao canto mais distante.

        Args:
            shape: Tupla ``(altura, largura)`` com dimensões positivas.
            cutoff: Raio real e finito, medido em bins de frequência.

        Returns:
            Novo array ``uint8`` binário com o shape solicitado.
        """
        rows, columns = self._validate_shape(shape)
        validated_cutoff = self._validate_cutoff(cutoff, rows, columns)
        row_coordinates = np.arange(rows) - rows // 2
        column_coordinates = np.arange(columns) - columns // 2
        distances = np.hypot(
            row_coordinates[:, np.newaxis],
            column_coordinates[np.newaxis, :],
        )
        return (distances <= validated_cutoff).astype(np.uint8)

    def ideal_high_pass_mask(
        self,
        shape: tuple[int, int],
        cutoff: float,
    ) -> np.ndarray:
        """Cria o complemento binário da máscara passa-baixa ideal.

        ``shape`` e ``cutoff`` seguem :meth:`ideal_low_pass_mask`; todo ponto
        satisfaz ``high_pass == 1 - low_pass``.
        """
        return 1 - self.ideal_low_pass_mask(shape, cutoff)

    def apply_mask(
        self,
        spectrum: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Multiplica um espectro por uma máscara binária de mesmo shape.

        O resultado permanece complexo, 2D e centralizado. A operação retorna
        um novo array e não modifica nenhuma das entradas.
        """
        self._validate_spectrum(spectrum)
        self._validate_mask(mask)
        if spectrum.shape != mask.shape:
            raise ValueError("mask deve possuir o mesmo shape de spectrum.")
        return spectrum * mask

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image deve ser um array NumPy.")
        if image.ndim != 2:
            raise ValueError("image deve ser uma imagem grayscale 2D.")
        if image.size == 0:
            raise ValueError("image não pode estar vazia.")
        if np.issubdtype(image.dtype, np.bool_) or not (
            np.issubdtype(image.dtype, np.integer)
            or np.issubdtype(image.dtype, np.floating)
        ):
            raise TypeError("image deve usar um dtype NumPy real numérico.")
        if not np.all(np.isfinite(image)):
            raise ValueError("image deve conter somente valores finitos.")

    @staticmethod
    def _validate_spectrum(spectrum: np.ndarray) -> None:
        if not isinstance(spectrum, np.ndarray):
            raise TypeError("spectrum deve ser um array NumPy.")
        if spectrum.ndim != 2:
            raise ValueError("spectrum deve ter duas dimensões.")
        if spectrum.size == 0:
            raise ValueError("spectrum não pode estar vazio.")
        if not np.issubdtype(spectrum.dtype, np.complexfloating):
            raise TypeError("spectrum deve usar um dtype NumPy complexo.")
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("spectrum deve conter somente valores finitos.")

    @staticmethod
    def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise TypeError("shape deve ser uma tupla (altura, largura).")
        if any(
            isinstance(dimension, (bool, np.bool_))
            or not isinstance(dimension, Integral)
            for dimension in shape
        ):
            raise TypeError("As dimensões de shape devem ser números inteiros.")
        if any(dimension <= 0 for dimension in shape):
            raise ValueError("As dimensões de shape devem ser positivas.")
        return int(shape[0]), int(shape[1])

    @staticmethod
    def _validate_cutoff(cutoff: float, rows: int, columns: int) -> float:
        if isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, Real):
            raise TypeError("cutoff deve ser um número real.")
        validated = float(cutoff)
        if not np.isfinite(validated):
            raise ValueError("cutoff deve ser finito.")
        maximum = float(np.hypot(rows // 2, columns // 2))
        if validated < 0 or validated > maximum:
            raise ValueError(f"cutoff deve pertencer ao intervalo [0, {maximum}].")
        return validated

    @staticmethod
    def _validate_mask(mask: np.ndarray) -> None:
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask deve ser um array NumPy.")
        if mask.ndim != 2:
            raise ValueError("mask deve ter duas dimensões.")
        if mask.size == 0:
            raise ValueError("mask não pode estar vazia.")
        if (
            np.issubdtype(mask.dtype, np.bool_)
            or not np.issubdtype(mask.dtype, np.number)
            or np.iscomplexobj(mask)
        ):
            raise TypeError("mask deve usar um dtype NumPy real numérico.")
        if not np.all(np.isfinite(mask)):
            raise ValueError("mask deve conter somente valores finitos.")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("mask deve conter somente os valores 0 e 1.")
