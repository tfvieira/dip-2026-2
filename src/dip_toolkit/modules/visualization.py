from __future__ import annotations

from collections.abc import Sequence
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
    ) -> tuple[Figure, Axes]:
        """Exibe uma imagem e retorna a figura e os eixos para composição.

        Imagens 2D devem usar ``channel_order="gray"``. Imagens coloridas 3D
        devem informar explicitamente ``"bgr"`` ou ``"rgb"``; BGR é convertido
        para RGB antes de ser mostrado pelo Matplotlib. A função não chama
        ``plt.show()``, permitindo compor e testar a figura retornada.

        Args:
            image: Imagem 2D em escala de cinza ou 3D com três canais.
            channel_order: Ordem dos canais da imagem.
            title: Título opcional para o eixo.
            ax: Eixo existente para composição. Quando omitido, cria um novo.

        Returns:
            A figura e o eixo que contêm a imagem.
        """

        self._validate_image(image)
        self._validate_channel_order(image, channel_order)

        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure

        rendered = (
            cv.cvtColor(image, cv.COLOR_BGR2RGB) if channel_order == "bgr" else image
        )
        if image.ndim == 2:
            ax.imshow(
                rendered, cmap="gray", vmin=self._vmin(image), vmax=self._vmax(image)
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
    ) -> tuple[Figure, np.ndarray]:
        """Exibe imagens lado a lado e retorna figura e eixos.

        Todas as imagens devem possuir a mesma dimensionalidade e ordem de canais
        indicada por ``channel_order``. Para uma imagem colorida carregada pelo
        OpenCV, informe ``channel_order="bgr"``.

        Args:
            images: Sequência não vazia de imagens 2D ou 3D com três canais.
            titles: Um título para cada imagem.
            channel_order: Ordem dos canais compartilhada pelas imagens.

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
    def _vmin(image: np.ndarray) -> float | None:
        return 0.0 if np.issubdtype(image.dtype, np.floating) else None

    @staticmethod
    def _vmax(image: np.ndarray) -> float | None:
        return 1.0 if np.issubdtype(image.dtype, np.floating) else None
