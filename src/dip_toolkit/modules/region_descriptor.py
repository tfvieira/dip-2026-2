"""Representação e descritores básicos de regiões segmentadas."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import NamedTuple

import cv2 as cv
import numpy as np

from .image_segmenter import ConnectedComponents


class RegionMoments(NamedTuple):
    """Momentos brutos e centrais mínimos de uma região binária."""

    m00: float
    m10: float
    m01: float
    mu20: float
    mu02: float
    mu11: float


class RegionDescriptor(NamedTuple):
    """Representação determinística de uma região de primeiro plano.

    ``contour`` tem shape ``(n, 1, 2)`` e coordenadas ``(x, y)``. A caixa é
    ``(x, y, width, height)``. A área é a contagem de pixels; o perímetro vem
    da fronteira extraída pelo OpenCV. A circularidade usa a área geométrica
    do contorno para manter a mesma convenção do perímetro. Contornos
    degenerados têm perímetro e circularidade iguais a 0.
    """

    region_id: int
    contour: np.ndarray
    area: float
    perimeter: float
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]
    aspect_ratio: float
    circularity: float
    moments: RegionMoments


class RegionDescriptorExtractor:
    """Extrai fronteiras e descritores de máscaras ou rótulos já segmentados.

    Não executa limiarização. Máscaras são arrays 2D ``uint8`` com 0 e 255;
    mapas de rótulos são arrays 2D inteiros não negativos com fundo igual a 0.
    """

    def describe_regions(
        self, regions: ConnectedComponents
    ) -> tuple[RegionDescriptor, ...]:
        """Descreve diretamente a saída de componentes conectados."""
        if not isinstance(regions, ConnectedComponents):
            raise TypeError("regions deve ser um resultado de componentes conectados.")
        self._validate_mask(regions.mask)
        self._validate_labels(regions.labels)
        if regions.mask.shape != regions.labels.shape:
            raise ValueError("mask e labels devem possuir o mesmo shape.")
        expected_ids = tuple(
            sorted(int(value) for value in np.unique(regions.labels) if value)
        )
        if regions.region_ids != expected_ids:
            raise ValueError("region_ids deve corresponder aos rótulos presentes.")
        return self.describe_labels(regions.labels, regions.region_ids)

    def describe_labels(
        self,
        labels: np.ndarray,
        region_ids: Sequence[int] | None = None,
    ) -> tuple[RegionDescriptor, ...]:
        """Descreve regiões de ``labels`` em ordem crescente de identificador."""
        labels = self._validate_labels(labels)
        available_ids = tuple(
            sorted(int(value) for value in np.unique(labels) if value)
        )
        ids = (
            available_ids
            if region_ids is None
            else self._validate_region_ids(region_ids)
        )
        if any(region_id not in available_ids for region_id in ids):
            raise ValueError("region_ids deve referenciar rótulos presentes em labels.")
        return tuple(
            self.describe_region(
                (labels == region_id).astype(np.uint8) * 255, region_id
            )
            for region_id in ids
        )

    def describe_mask(self, mask: np.ndarray) -> tuple[RegionDescriptor, ...]:
        """Descreve os componentes 8-conectados de uma máscara binária pronta."""
        mask = self._validate_mask(mask)
        total, labels = cv.connectedComponents(mask, connectivity=8, ltype=cv.CV_32S)
        return self.describe_labels(labels, tuple(range(1, total)))

    def describe_region(self, mask: np.ndarray, region_id: int = 1) -> RegionDescriptor:
        """Descreve uma região binária única, não vazia."""
        mask = self._validate_mask(mask)
        region_id = self._validate_region_id(region_id)
        if not np.any(mask):
            raise ValueError("mask deve conter ao menos um pixel de primeiro plano.")
        total_components, _ = cv.connectedComponents(
            mask, connectivity=8, ltype=cv.CV_32S
        )
        if total_components != 2:
            raise ValueError("mask deve conter exatamente uma região conectada.")
        contour = self.extract_contour(mask)
        values = cv.moments(mask, binaryImage=True)
        area = float(values["m00"])
        centroid = (float(values["m10"] / area), float(values["m01"] / area))
        x, y, width, height = cv.boundingRect(contour)
        perimeter = 0.0 if len(contour) < 2 else float(cv.arcLength(contour, True))
        contour_area = float(cv.contourArea(contour))
        circularity = (
            0.0 if perimeter == 0 else float(4 * np.pi * contour_area / perimeter**2)
        )
        return RegionDescriptor(
            region_id=region_id,
            contour=contour,
            area=area,
            perimeter=perimeter,
            centroid=centroid,
            bounding_box=(int(x), int(y), int(width), int(height)),
            aspect_ratio=float(width / height),
            circularity=circularity,
            moments=RegionMoments(
                m00=area,
                m10=float(values["m10"]),
                m01=float(values["m01"]),
                mu20=float(values["mu20"]),
                mu02=float(values["mu02"]),
                mu11=float(values["mu11"]),
            ),
        )

    def extract_contour(self, mask: np.ndarray) -> np.ndarray:
        """Extrai o maior contorno externo de uma máscara binária não vazia."""
        mask = self._validate_mask(mask)
        contours, _ = cv.findContours(
            mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
        )
        if not contours:
            raise ValueError("mask deve conter ao menos um contorno.")
        return max(contours, key=cv.contourArea).copy()

    def plot_regions(
        self, mask: np.ndarray, descriptors: Sequence[RegionDescriptor]
    ) -> tuple[object, object]:
        """Exibe máscara, contornos, centroides e caixas com Matplotlib."""
        mask = self._validate_mask(mask)
        if not isinstance(descriptors, Sequence):
            raise TypeError("descriptors deve ser uma sequência de descritores.")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(mask, cmap="gray", vmin=0, vmax=255)
        for descriptor in descriptors:
            if not isinstance(descriptor, RegionDescriptor):
                raise TypeError("descriptors deve conter RegionDescriptor.")
            points = descriptor.contour.reshape(-1, 2)
            axis.plot(points[:, 0], points[:, 1], color="lime", linewidth=1.5)
            x, y, width, height = descriptor.bounding_box
            axis.add_patch(
                Rectangle((x, y), width, height, fill=False, edgecolor="red")
            )
            axis.plot(*descriptor.centroid, marker="+", color="yellow", markersize=10)
            axis.text(x, y - 1, str(descriptor.region_id), color="yellow")
        axis.set_title("Regiões, contornos e descritores")
        axis.axis("off")
        return figure, axis

    @staticmethod
    def _validate_mask(mask: np.ndarray) -> np.ndarray:
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask deve ser um array NumPy.")
        if mask.ndim != 2 or mask.size == 0:
            raise ValueError("mask deve ser uma máscara binária 2D não vazia.")
        if mask.dtype != np.dtype(np.uint8):
            raise TypeError("mask deve possuir dtype uint8.")
        if not np.all((mask == 0) | (mask == 255)):
            raise ValueError("mask deve conter somente os valores 0 e 255.")
        return mask

    @staticmethod
    def _validate_labels(labels: np.ndarray) -> np.ndarray:
        if not isinstance(labels, np.ndarray):
            raise TypeError("labels deve ser um array NumPy.")
        if labels.ndim != 2 or labels.size == 0:
            raise ValueError("labels deve ser um mapa 2D não vazio.")
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("labels deve possuir dtype inteiro.")
        if np.any(labels < 0):
            raise ValueError("labels não pode conter valores negativos.")
        return labels

    @staticmethod
    def _validate_region_ids(region_ids: Sequence[int]) -> tuple[int, ...]:
        if isinstance(region_ids, (str, bytes)) or not isinstance(region_ids, Sequence):
            raise TypeError("region_ids deve ser uma sequência de inteiros.")
        values = tuple(
            RegionDescriptorExtractor._validate_region_id(value) for value in region_ids
        )
        if tuple(sorted(values)) != values or len(set(values)) != len(values):
            raise ValueError(
                "region_ids deve estar em ordem crescente e sem repetição."
            )
        return values

    @staticmethod
    def _validate_region_id(region_id: int) -> int:
        if isinstance(region_id, (bool, np.bool_)) or not isinstance(
            region_id, Integral
        ):
            raise TypeError("region_id deve ser um inteiro positivo.")
        if region_id <= 0:
            raise ValueError("region_id deve ser um inteiro positivo.")
        return int(region_id)
