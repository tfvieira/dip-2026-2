from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Literal

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ChannelOrder = Literal["gray", "bgr", "rgb"]


class Visualization:
    """Cria visualizações didáticas de arrays de imagem com Matplotlib."""

    def plot_histogram(
        self,
        counts: np.ndarray,
        bin_edges: np.ndarray | Sequence[Real] | None = None,
        *,
        ax: Axes | None = None,
        title: str | None = "Image Histogram",
        label: str | None = None,
        color: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Plota contagens de histograma sem chamar plt.show().

        Para compatibilidade com a API anterior, counts também pode ter shape
        (N, 1). Quando bin_edges é informado, a linha usa o centro de cada bin
        como coordenada horizontal. O eixo opcional permite compor vários
        histogramas na mesma figura.

        Args:
            counts: Contagens não negativas em um array 1D ou (N, 1).
            bin_edges: Bordas 1D, finitas e estritamente crescentes, de tamanho
                N + 1. Quando omitidas, usa os índices das contagens.
            ax: Eixo Matplotlib existente. Quando omitido, cria um novo.
            title: Título opcional do eixo.
            label: Rótulo opcional para a linha e a legenda.
            color: Cor opcional aceita pelo Matplotlib.

        Returns:
            Figura e eixo que contêm a linha do histograma.
        """

        validated_counts = self._validate_histogram_counts(counts)
        x_values = self._resolve_histogram_x(validated_counts, bin_edges)
        if ax is not None and not isinstance(ax, Axes):
            raise TypeError("ax deve ser um eixo Matplotlib.")
        if title is not None and not isinstance(title, str):
            raise TypeError("title deve ser uma string ou None.")
        if label is not None and not isinstance(label, str):
            raise TypeError("label deve ser uma string ou None.")

        if ax is None:
            figure, axis = plt.subplots()
        else:
            figure, axis = ax.figure, ax
        axis.plot(x_values, validated_counts, label=label, color=color)
        if title is not None:
            axis.set_title(title)
        axis.set_xlabel("Intensidade")
        axis.set_ylabel("Frequência")
        if label is not None:
            axis.legend()
        return figure, axis

    def show_image(
        self,
        image: np.ndarray,
        *,
        channel_order: ChannelOrder = "gray",
        title: str | None = None,
        ax: Axes | None = None,
        value_range: tuple[Real, Real] | None = None,
    ) -> tuple[Figure, Axes]:
        """Exibe uma imagem e retorna a figura e os eixos para composição.

        Imagens 2D devem usar channel_order="gray". Imagens coloridas 3D devem
        informar "bgr" ou "rgb"; BGR é convertido para RGB antes da exibição.
        A função não chama plt.show().

        Args:
            image: Imagem 2D em escala de cinza ou 3D com três canais.
            channel_order: Ordem dos canais da imagem.
            title: Título opcional para o eixo.
            ax: Eixo existente para composição. Quando omitido, cria um novo.
            value_range: Faixa opcional (mínimo, máximo) para imagens float.

        Returns:
            A figura e o eixo que contêm a imagem.
        """

        self._validate_image(image)
        self._validate_channel_order(image, channel_order)
        resolved_range = self._resolve_value_range(image, value_range)

        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure

        rendered = self._render_for_matplotlib(
            image,
            channel_order,
            resolved_range,
        )
        if image.ndim == 2:
            if resolved_range is None:
                ax.imshow(rendered, cmap="gray")
            else:
                ax.imshow(
                    rendered,
                    cmap="gray",
                    vmin=resolved_range[0],
                    vmax=resolved_range[1],
                )
        else:
            ax.imshow(rendered)
        if title is not None:
            ax.set_title(title)
        ax.axis("off")
        return figure, ax

    def compare_images(
        self,
        images: Sequence[np.ndarray],
        titles: Sequence[str],
        *,
        channel_order: ChannelOrder = "gray",
        value_range: tuple[Real, Real] | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """Exibe imagens lado a lado e retorna figura e eixos.

        Args:
            images: Sequência não vazia de imagens 2D ou 3D com três canais.
            titles: Um título para cada imagem.
            channel_order: Ordem dos canais compartilhada pelas imagens.
            value_range: Faixa opcional compartilhada para imagens float.

        Returns:
            Figura Matplotlib e array unidimensional de eixos.
        """

        if isinstance(images, (str, bytes)) or not isinstance(images, Sequence):
            raise TypeError("images deve ser uma sequência de arrays NumPy.")
        if not images:
            raise ValueError("images não pode estar vazia.")
        if len(images) != len(titles):
            raise ValueError("images e titles devem possuir o mesmo tamanho.")
        if not all(isinstance(title, str) for title in titles):
            raise TypeError("titles deve conter somente strings.")

        for image in images:
            self._validate_image(image)
            self._validate_channel_order(image, channel_order)

        figure, axes = plt.subplots(1, len(images), squeeze=False)
        flattened_axes = axes.ravel()
        for image, title, axis in zip(images, titles, flattened_axes, strict=True):
            self.show_image(
                image,
                channel_order=channel_order,
                title=title,
                ax=axis,
                value_range=value_range,
            )
        figure.tight_layout()
        return figure, flattened_axes

    @staticmethod
    def _validate_histogram_counts(counts: np.ndarray) -> np.ndarray:
        if not isinstance(counts, np.ndarray):
            raise TypeError("counts deve ser um array NumPy.")
        if counts.ndim == 2 and counts.shape[1:] == (1,):
            counts = counts[:, 0]
        if counts.ndim != 1:
            raise ValueError("counts deve ser unidimensional ou possuir shape (N, 1).")
        if counts.size == 0:
            raise ValueError("counts não pode estar vazio.")
        if np.issubdtype(counts.dtype, np.bool_) or not (
            np.issubdtype(counts.dtype, np.integer)
            or np.issubdtype(counts.dtype, np.floating)
        ):
            raise TypeError("counts deve usar dtype inteiro ou float.")
        if not np.all(np.isfinite(counts)):
            raise ValueError("counts deve conter somente valores finitos.")
        if np.any(counts < 0):
            raise ValueError("counts não pode conter valores negativos.")
        return counts

    @staticmethod
    def _resolve_histogram_x(
        counts: np.ndarray,
        bin_edges: np.ndarray | Sequence[Real] | None,
    ) -> np.ndarray:
        if bin_edges is None:
            return np.arange(counts.size)
        try:
            edges = np.asarray(bin_edges)
        except (TypeError, ValueError) as error:
            raise TypeError("bin_edges deve ser uma sequência numérica.") from error
        if edges.ndim != 1:
            raise ValueError("bin_edges deve ser unidimensional.")
        if edges.size != counts.size + 1:
            raise ValueError("bin_edges deve possuir len(counts) + 1 valores.")
        if np.issubdtype(edges.dtype, np.bool_) or not np.issubdtype(
            edges.dtype, np.number
        ):
            raise TypeError("bin_edges deve conter números.")
        if np.iscomplexobj(edges):
            raise TypeError("bin_edges deve conter números reais.")
        if not np.all(np.isfinite(edges)):
            raise ValueError("bin_edges deve conter somente valores finitos.")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("bin_edges deve ser estritamente crescente.")
        return (edges[:-1] + edges[1:]) / 2

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
    def _validate_channel_order(image: np.ndarray, channel_order: ChannelOrder) -> None:
        if channel_order not in {"gray", "bgr", "rgb"}:
            raise ValueError("channel_order deve ser 'gray', 'bgr' ou 'rgb'.")
        if image.ndim == 2 and channel_order != "gray":
            raise ValueError("Imagens 2D devem usar channel_order='gray'.")
        if image.ndim == 3 and channel_order == "gray":
            raise ValueError(
                "Imagens coloridas devem usar channel_order='bgr' ou 'rgb'."
            )

    @staticmethod
    def _resolve_value_range(
        image: np.ndarray,
        value_range: tuple[Real, Real] | None,
    ) -> tuple[float, float] | None:
        if value_range is None:
            return None
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
        if np.any(image < lower_float) or np.any(image > upper_float):
            raise ValueError("image contém valores fora de value_range.")
        return lower_float, upper_float

    @staticmethod
    def _render_for_matplotlib(
        image: np.ndarray,
        channel_order: ChannelOrder,
        value_range: tuple[float, float] | None,
    ) -> np.ndarray:
        rendered = (
            cv.cvtColor(image, cv.COLOR_BGR2RGB) if channel_order == "bgr" else image
        )
        if image.ndim != 3 or not np.issubdtype(image.dtype, np.floating):
            return rendered
        if value_range is None:
            lower = float(np.min(rendered))
            upper = float(np.max(rendered))
            if 0.0 <= lower and upper <= 1.0:
                return rendered
            if lower == upper:
                raise ValueError(
                    "Imagens coloridas float constantes fora de [0, 1] "
                    "devem informar value_range."
                )
        else:
            lower, upper = value_range
        return (rendered - lower) / (upper - lower)
