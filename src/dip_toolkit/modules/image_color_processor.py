"""Operações didáticas com imagens coloridas e convenções explícitas de canais."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, NamedTuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from .intensity_transformer import IntensityTransformer

ColorOrder = Literal["bgr", "rgb"]
ColorSpace = Literal["bgr", "rgb", "hsv", "ycrcb", "lab", "gray"]
ChannelOperation = Literal["negative", "log", "gamma", "piecewise"]

_COLOR_ORDERS = {"bgr", "rgb"}
_COLOR_SPACES = {"bgr", "rgb", "hsv", "ycrcb", "lab", "gray"}


class ChannelSet(NamedTuple):
    """Canais separados junto à ordem necessária para recombiná-los.

    Attributes:
        channels: Canais 2D na mesma ordem declarada em ``channel_order``.
        channel_order: Convenção explícita ``"bgr"`` ou ``"rgb"``.
    """

    channels: tuple[np.ndarray, np.ndarray, np.ndarray]
    channel_order: ColorOrder


class ColorImageProcessor:
    """Calcula transformações e distribuições em imagens coloridas uint8.

    Imagens coloridas usam shape ``(height, width, 3)`` e dtype ``uint8``.
    A ordem BGR corresponde ao carregamento padrão do OpenCV; RGB é a ordem
    usual de exibição pelo Matplotlib. HSV usa H em ``[0, 179]`` e os outros
    canais em ``[0, 255]``; YCrCb e Lab usam três canais ``uint8`` em
    ``[0, 255]``. Grayscale usa shape ``(height, width)`` e dtype ``uint8``.
    Métodos de cálculo não modificam a entrada.
    """

    def split_channels(
        self, image: np.ndarray, channel_order: ColorOrder
    ) -> ChannelSet:
        """Separa uma imagem BGR ou RGB em canais 2D copiáveis.

        Args:
            image: Imagem colorida uint8 de shape ``(height, width, 3)``.
            channel_order: Ordem declarada dos canais na entrada.

        Returns:
            ChannelSet com canais na ordem indicada e metadado para recombinação.
        """
        order = self._validate_channel_order(channel_order)
        self._validate_color_image(image, order)
        return ChannelSet(tuple(image[..., index].copy() for index in range(3)), order)

    def combine_channels(self, channels: ChannelSet) -> np.ndarray:
        """Combina canais separados em uma nova imagem colorida uint8.

        Args:
            channels: Canais e ordem retornados por :meth:`split_channels`.

        Returns:
            Imagem com shape ``(height, width, 3)`` na ordem declarada.
        """
        if not isinstance(channels, ChannelSet):
            raise TypeError("channels deve ser um ChannelSet.")
        self._validate_channel_order(channels.channel_order)
        if len(channels.channels) != 3:
            raise ValueError("channels deve conter exatamente três canais.")
        first, second, third = channels.channels
        for channel in channels.channels:
            if not isinstance(channel, np.ndarray):
                raise TypeError("Cada canal deve ser um array NumPy.")
            if channel.ndim != 2 or channel.size == 0:
                raise ValueError(
                    "Cada canal deve ser uma imagem grayscale 2D não vazia."
                )
        if second.shape != first.shape or third.shape != first.shape:
            raise ValueError("Todos os canais devem possuir o mesmo shape.")
        if second.dtype != first.dtype or third.dtype != first.dtype:
            raise TypeError("Todos os canais devem possuir o mesmo dtype.")
        for channel in channels.channels:
            self._validate_gray_image(channel)
        return np.stack(channels.channels, axis=-1)

    def convert(
        self,
        image: np.ndarray,
        source: ColorSpace,
        destination: ColorSpace,
    ) -> np.ndarray:
        """Converte uma imagem uint8 entre espaços de cor documentados.

        Conversões partem da convenção indicada em ``source`` e retornam a
        convenção indicada em ``destination``. A conversão RGB↔BGR é exata;
        conversões que passam por HSV, YCrCb ou Lab podem variar levemente em
        uma ida e volta por arredondamento do OpenCV.
        """
        source_space = self._validate_color_space(source, "source")
        destination_space = self._validate_color_space(destination, "destination")
        self._validate_space_image(image, source_space)
        if source_space == destination_space:
            return image.copy()

        bgr = self._to_bgr(image, source_space)
        return self._from_bgr(bgr, destination_space)

    def channel_histograms(
        self,
        image: np.ndarray,
        channel_order: ColorOrder,
    ) -> dict[str, np.ndarray]:
        """Calcula histogramas de 256 níveis na ordem BGR ou RGB declarada.

        Returns:
            Dicionário ordenado com chaves ``b``, ``g``, ``r`` ou ``r``,
            ``g``, ``b``. Cada histograma tem shape ``(256,)`` e a soma igual
            ao número de pixels de seu canal.
        """
        order = self._validate_channel_order(channel_order)
        self._validate_color_image(image, order)
        return {
            channel: np.bincount(image[..., index].ravel(), minlength=256)
            for index, channel in enumerate(order)
        }

    def apply_channel_operation(
        self,
        image: np.ndarray,
        channel_order: ColorOrder,
        operation: ChannelOperation,
        **parameters: object,
    ) -> np.ndarray:
        """Aplica uma transformação de intensidade independentemente por canal.

        Args:
            image: Imagem BGR ou RGB uint8.
            channel_order: Ordem declarada dos canais.
            operation: ``negative``, ``log``, ``gamma`` ou ``piecewise``.
            **parameters: Parâmetros aceitos pela transformação correspondente.

        Returns:
            Nova imagem com o mesmo shape e dtype da entrada.
        """
        order = self._validate_channel_order(channel_order)
        self._validate_color_image(image, order)
        if operation not in {"negative", "log", "gamma", "piecewise"}:
            raise ValueError(
                "operation deve ser 'negative', 'log', 'gamma' ou 'piecewise'."
            )
        transformer = IntensityTransformer()
        transformed_channels: list[np.ndarray] = []
        for index in range(3):
            channel = image[..., index]
            if operation == "negative":
                transformed = transformer.negative(channel, **parameters)
            elif operation == "log":
                transformed = transformer.log_transform(channel, **parameters)
            elif operation == "gamma":
                transformed = transformer.gamma_transform(channel, **parameters)
            else:
                transformed = transformer.piecewise_linear(channel, **parameters)
            transformed_channels.append(transformed)
        return np.stack(transformed_channels, axis=-1)

    def plot_channel_histograms(
        self,
        histograms: Mapping[str, np.ndarray],
        channel_order: ColorOrder,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plota histogramas calculados, sem executar ``plt.show()``."""
        order = self._validate_channel_order(channel_order)
        if list(histograms) != list(order):
            raise ValueError("histograms deve usar as chaves na ordem declarada.")
        figure, axis = plt.subplots(figsize=(9, 4))
        colors = {"b": "blue", "g": "green", "r": "red"}
        for channel in order:
            histogram = histograms[channel]
            if not isinstance(histogram, np.ndarray) or histogram.shape != (256,):
                raise ValueError("Cada histograma deve ser um array com shape (256,).")
            axis.plot(histogram, color=colors[channel], label=channel.upper())
        axis.set(
            title=f"Histogramas por canal ({order.upper()})",
            xlabel="Intensidade",
            ylabel="Contagem",
            xlim=(0, 255),
        )
        axis.legend()
        axis.grid(alpha=0.25)
        return figure, axis

    def plot_color_cube(
        self,
        image: np.ndarray,
        channel_order: ColorOrder,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plota valores no cubo RGB retornando figura e eixo Matplotlib."""
        order = self._validate_channel_order(channel_order)
        self._validate_color_image(image, order)
        rgb = image if order == "rgb" else self.convert(image, "bgr", "rgb")
        figure = plt.figure(figsize=(7, 7))
        axis = figure.add_subplot(111, projection="3d")
        pixels = rgb.reshape(-1, 3)
        axis.scatter(
            pixels[:, 0],
            pixels[:, 1],
            pixels[:, 2],
            c=pixels / 255.0,
            marker=".",
        )
        axis.set(xlabel="Red", ylabel="Green", zlabel="Blue", title="RGB color cube")
        return figure, axis

    def plot_rgb_histograms(self, image: np.ndarray) -> None:
        """Mantém a API legada para histogramas de uma imagem BGR uint8.

        O nome foi preservado por compatibilidade. Como no comportamento
        original, a entrada é tratada como BGR (a convenção do OpenCV), a
        figura é exibida com ``plt.show()`` e nenhum valor é retornado.
        Para composição sem efeitos visuais, use :meth:`channel_histograms` e
        :meth:`plot_channel_histograms`.
        """
        histograms = self.channel_histograms(image, "bgr")
        self.plot_channel_histograms(histograms, "bgr")
        plt.show()

    def convert_color_space(self, image: np.ndarray, conversion: str) -> np.ndarray:
        """Mantém a API legada de conversão partindo de uma imagem BGR.

        As chaves históricas usam o prefixo ``rgb_to_``, mas a implementação
        original empregava códigos ``BGR`` do OpenCV. Esse comportamento é
        mantido; para uma convenção explícita, use :meth:`convert`.
        """
        destinations: dict[str, ColorSpace] = {
            "rgb_to_hsv": "hsv",
            "rgb_to_ycrcb": "ycrcb",
            "rgb_to_lab": "lab",
            "rgb_to_gray": "gray",
        }
        if conversion not in destinations:
            supported = list(destinations)
            raise ValueError(
                f"Unsupported conversion: {conversion}. Supported: {supported}"
            )
        return self.convert(image, "bgr", destinations[conversion])

    def rgb_to_cmyk(self, image: np.ndarray) -> np.ndarray:
        """Mantém a API legada de conversão de imagem BGR uint8 para CMYK.

        A saída tem shape ``(height, width, 4)``, dtype ``uint8`` e canais
        CMYK. Embora o nome histórico mencione RGB, a entrada preserva a
        convenção BGR usada pelo OpenCV na implementação original.
        """
        self._validate_color_image(image, "bgr")
        blue, green, red = (
            image[..., index].astype(np.float64) / 255.0 for index in range(3)
        )
        black = 1.0 - np.maximum.reduce((red, green, blue))
        denominator = 1.0 - black + 1e-5
        cyan = (1.0 - red - black) / denominator
        magenta = (1.0 - green - black) / denominator
        yellow = (1.0 - blue - black) / denominator
        cmyk = np.stack((cyan, magenta, yellow, black), axis=-1)
        return (cmyk * 255.0).astype(np.uint8)

    def plot_rgb_3d_cube(self, image: np.ndarray) -> None:
        """Mantém a API legada que exibe o cubo de cores de uma imagem BGR."""
        self.plot_color_cube(image, "bgr")
        plt.show()

    @staticmethod
    def _validate_channel_order(channel_order: ColorOrder) -> ColorOrder:
        if not isinstance(channel_order, str) or channel_order not in _COLOR_ORDERS:
            raise ValueError("channel_order deve ser 'bgr' ou 'rgb'.")
        return channel_order

    @staticmethod
    def _validate_color_space(value: ColorSpace, name: str) -> ColorSpace:
        if not isinstance(value, str) or value not in _COLOR_SPACES:
            raise ValueError(
                f"{name} deve ser 'bgr', 'rgb', 'hsv', 'ycrcb', 'lab' ou 'gray'."
            )
        return value

    @staticmethod
    def _validate_gray_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image deve ser um array NumPy.")
        if image.ndim != 2 or image.size == 0:
            raise ValueError("image deve ser uma imagem grayscale 2D não vazia.")
        if image.dtype != np.dtype(np.uint8):
            raise TypeError("image deve possuir dtype uint8.")

    @classmethod
    def _validate_color_image(cls, image: np.ndarray, _: ColorOrder) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image deve ser um array NumPy.")
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("image deve ter shape (height, width, 3) e não ser vazia.")
        if image.dtype != np.dtype(np.uint8):
            raise TypeError("image deve possuir dtype uint8.")

    @classmethod
    def _validate_space_image(cls, image: np.ndarray, space: ColorSpace) -> None:
        if space == "gray":
            cls._validate_gray_image(image)
            return
        cls._validate_color_image(image, "bgr")

    @staticmethod
    def _to_bgr(image: np.ndarray, source: ColorSpace) -> np.ndarray:
        conversions = {
            "bgr": None,
            "rgb": cv.COLOR_RGB2BGR,
            "hsv": cv.COLOR_HSV2BGR,
            "ycrcb": cv.COLOR_YCrCb2BGR,
            "lab": cv.COLOR_Lab2BGR,
            "gray": cv.COLOR_GRAY2BGR,
        }
        conversion = conversions[source]
        return image.copy() if conversion is None else cv.cvtColor(image, conversion)

    @staticmethod
    def _from_bgr(image: np.ndarray, destination: ColorSpace) -> np.ndarray:
        conversions = {
            "bgr": None,
            "rgb": cv.COLOR_BGR2RGB,
            "hsv": cv.COLOR_BGR2HSV,
            "ycrcb": cv.COLOR_BGR2YCrCb,
            "lab": cv.COLOR_BGR2Lab,
            "gray": cv.COLOR_BGR2GRAY,
        }
        conversion = conversions[destination]
        return image.copy() if conversion is None else cv.cvtColor(image, conversion)
