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

    def __init__(self):
        pass

    def plot_histogram(self, hist):
        plt.figure()
        plt.plot(hist)
        plt.title("Image Histogram")
        plt.show()

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

        Imagens 2D devem usar ``channel_order="gray"``. Imagens coloridas 3D
        devem informar explicitamente ``"bgr"`` ou ``"rgb"``; BGR é convertido
        para RGB antes de ser mostrado pelo Matplotlib. Para imagens float, uma
        faixa explícita permite visualizar valores em ``[-1, 1]`` sem supor
        ``[0, 1]``. A função não chama ``plt.show()``, permitindo compor e testar
        a figura retornada.

        Args:
            image: Imagem 2D em escala de cinza ou 3D com três canais.
            channel_order: Ordem dos canais da imagem.
            title: Título opcional para o eixo.
            ax: Eixo existente para composição. Quando omitido, cria um novo.
            value_range: Faixa opcional ``(mínimo, máximo)`` para exibição de
                imagens float. Valores fora dela são rejeitados.

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

        Todas as imagens devem possuir a mesma dimensionalidade e ordem de canais
        indicada por ``channel_order``. Para uma imagem colorida carregada pelo
        OpenCV, informe ``channel_order="bgr"``.

        Args:
            images: Sequência não vazia de imagens 2D ou 3D com três canais.
            titles: Um título para cada imagem.
            channel_order: Ordem dos canais compartilhada pelas imagens.
            value_range: Faixa opcional compartilhada para exibição de imagens
                float.

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
