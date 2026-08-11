from __future__ import annotations

from numbers import Integral, Real
from typing import Literal, NamedTuple

import numpy as np

ColorChannelOrder = Literal["bgr", "rgb"]
HistogramRange = tuple[Real, Real]


class HistogramResult(NamedTuple):
    """Contagens e bordas produzidas por numpy.histogram.

    Attributes:
        counts: Array int64 unidimensional com uma contagem por bin.
        bin_edges: Array float64 unidimensional com bins + 1 bordas.
    """

    counts: np.ndarray
    bin_edges: np.ndarray


class ImageAnalysis:
    """Calcula histogramas e distribuições acumuladas sem criar figuras."""

    def compute_grayscale_histogram(
        self,
        image: np.ndarray,
        *,
        bins: int = 256,
        value_range: HistogramRange | None = None,
    ) -> HistogramResult:
        """Calcula o histograma de uma imagem grayscale 2D.

        Quando value_range é omitido, a faixa cobre o menor e o maior valor da
        imagem; imagens constantes recebem uma faixa unitária centrada no valor.
        A faixa explícita deve conter todos os pixels.

        Args:
            image: Imagem grayscale 2D com dtype inteiro ou float.
            bins: Quantidade inteira e positiva de bins.
            value_range: Faixa opcional (mínimo, máximo) do histograma.

        Returns:
            HistogramResult com counts int64 de shape (bins,) e bin_edges
            float64 de shape (bins + 1,).
        """

        self._validate_numeric_image(image)
        if image.ndim != 2:
            raise ValueError(
                "compute_grayscale_histogram aceita somente imagens grayscale 2D."
            )
        validated_bins = self._validate_bins(bins)
        resolved_range = self._resolve_histogram_range(image, value_range)
        counts, bin_edges = np.histogram(
            image,
            bins=validated_bins,
            range=resolved_range,
        )
        return HistogramResult(
            counts.astype(np.int64, copy=False),
            bin_edges.astype(np.float64, copy=False),
        )

    def compute_color_histograms(
        self,
        image: np.ndarray,
        *,
        channel_order: ColorChannelOrder,
        bins: int = 256,
        value_range: HistogramRange | None = None,
    ) -> dict[str, HistogramResult]:
        """Calcula um histograma identificado para cada canal colorido.

        Args:
            image: Imagem 3D com exatamente três canais.
            channel_order: Ordem explícita dos canais, bgr ou rgb.
            bins: Quantidade inteira e positiva de bins por canal.
            value_range: Faixa opcional compartilhada pelos três canais.

        Returns:
            Dicionário determinístico, rotulado conforme channel_order.
        """

        self._validate_numeric_image(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "compute_color_histograms aceita somente imagens 3D com três canais."
            )
        if channel_order not in {"bgr", "rgb"}:
            raise ValueError("channel_order deve ser 'bgr' ou 'rgb'.")

        validated_bins = self._validate_bins(bins)
        resolved_range = self._resolve_histogram_range(image, value_range)
        results: dict[str, HistogramResult] = {}
        for index, channel in enumerate(channel_order):
            counts, bin_edges = np.histogram(
                image[..., index],
                bins=validated_bins,
                range=resolved_range,
            )
            results[channel] = HistogramResult(
                counts.astype(np.int64, copy=False),
                bin_edges.astype(np.float64, copy=False),
            )
        return results

    def compute_cdf(
        self,
        histogram: np.ndarray,
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        """Calcula a distribuição acumulada de um histograma 1D.

        Por padrão, o resultado float64 é normalizado em [0, 1]. Histogramas
        com soma zero retornam zeros. Sem normalização, retorna somas acumuladas.

        Args:
            histogram: Contagens inteiras ou float em um array 1D.
            normalize: Se verdadeiro, divide a CDF pela soma das contagens.

        Returns:
            CDF float64, não decrescente e com o mesmo shape da entrada.
        """

        if not isinstance(histogram, np.ndarray):
            raise TypeError("histogram deve ser um array NumPy.")
        if histogram.ndim != 1:
            raise ValueError("histogram deve ser unidimensional.")
        if np.issubdtype(histogram.dtype, np.bool_) or not (
            np.issubdtype(histogram.dtype, np.integer)
            or np.issubdtype(histogram.dtype, np.floating)
        ):
            raise TypeError("histogram deve usar dtype inteiro ou float.")
        if not np.all(np.isfinite(histogram)):
            raise ValueError("histogram deve conter somente valores finitos.")
        if np.any(histogram < 0):
            raise ValueError("histogram não pode conter contagens negativas.")
        if not isinstance(normalize, (bool, np.bool_)):
            raise TypeError("normalize deve ser booleano.")

        cdf = np.cumsum(histogram, dtype=np.float64)
        if bool(normalize):
            total = cdf[-1] if cdf.size else 0.0
            if total == 0.0:
                return np.zeros(histogram.shape, dtype=np.float64)
            cdf /= total
        return cdf

    def compute_histogram(self, image: np.ndarray) -> np.ndarray:
        """Mantém a API legada para histogramas grayscale de 256 bins.

        O retorno preserva o shape legado (256, 1). Imagens coloridas são
        rejeitadas; use compute_color_histograms para evitar seleção silenciosa.
        """

        if isinstance(image, np.ndarray) and image.ndim == 3:
            raise ValueError(
                "compute_histogram não aceita imagem colorida; use "
                "compute_color_histograms."
            )
        value_range: HistogramRange | None = None
        if isinstance(image, np.ndarray) and image.dtype == np.dtype(np.uint8):
            value_range = (0, 256)
        result = self.compute_grayscale_histogram(
            image,
            bins=256,
            value_range=value_range,
        )
        return result.counts.reshape(-1, 1)

    @staticmethod
    def _validate_numeric_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image deve ser um array NumPy.")
        if image.ndim not in {2, 3}:
            raise ValueError("image deve ter duas ou três dimensões.")
        if any(dimension <= 0 for dimension in image.shape):
            raise ValueError("image não pode possuir dimensões vazias.")
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError("Imagens coloridas devem possuir exatamente três canais.")
        if np.issubdtype(image.dtype, np.bool_) or not (
            np.issubdtype(image.dtype, np.integer)
            or np.issubdtype(image.dtype, np.floating)
        ):
            raise TypeError("image deve usar um dtype NumPy inteiro ou float.")
        if not np.all(np.isfinite(image)):
            raise ValueError("image deve conter somente valores finitos.")

    @staticmethod
    def _validate_bins(bins: int) -> int:
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, Integral):
            raise TypeError("bins deve ser um número inteiro.")
        if bins <= 0:
            raise ValueError("bins deve ser maior que zero.")
        return int(bins)

    @staticmethod
    def _resolve_histogram_range(
        image: np.ndarray,
        value_range: HistogramRange | None,
    ) -> tuple[float, float]:
        if value_range is None:
            lower = float(np.min(image))
            upper = float(np.max(image))
            if lower == upper:
                return lower - 0.5, upper + 0.5
            return lower, upper

        if not isinstance(value_range, tuple) or len(value_range) != 2:
            raise TypeError("value_range deve ser uma tupla (mínimo, máximo).")
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in value_range
        ):
            raise TypeError("Os limites de value_range devem ser números reais.")
        lower, upper = (float(value) for value in value_range)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("Os limites de value_range devem ser finitos.")
        if lower >= upper:
            raise ValueError("value_range deve possuir mínimo menor que máximo.")
        if np.any(image < lower) or np.any(image > upper):
            raise ValueError("image contém valores fora de value_range.")
        return lower, upper
