"""Operações de morfologia matemática para máscaras binárias."""

from __future__ import annotations

from numbers import Integral
from typing import Literal

import cv2 as cv
import numpy as np

StructuringElementShape = Literal["rectangle", "ellipse", "cross"]
BorderPolicy = Literal["constant", "replicate"]

_STRUCTURING_ELEMENT_SHAPES = {"rectangle", "ellipse", "cross"}
_BORDER_POLICIES = {"constant", "replicate"}


class MorphologyProcessor:
    """Aplica morfologia binária a máscaras NumPy ``uint8``.

    As máscaras aceitas têm shape ``(height, width)``, dtype ``uint8`` e usam
    somente ``0`` para fundo e ``255`` para primeiro plano. As operações
    retornam uma nova máscara com o mesmo shape, dtype e conjunto de valores;
    a entrada nunca é modificada. A borda ``constant`` considera pixels fora
    da imagem como fundo, enquanto ``replicate`` repete o pixel mais próximo.
    """

    def create_structuring_element(
        self,
        shape: StructuringElementShape,
        size: tuple[int, int] = (3, 3),
    ) -> np.ndarray:
        """Cria elemento estruturante binário ``uint8`` com dimensões ímpares.

        Args:
            shape: Formato ``rectangle``, ``ellipse`` ou ``cross``.
            size: Tupla ``(width, height)`` positiva, com ambos valores ímpares.

        Returns:
            Array 2D ``uint8`` com valores ``0`` e ``1``.
        """
        element_shape = self._validate_element_shape(shape)
        width, height = self._validate_size(size)
        cv_shapes = {
            "rectangle": cv.MORPH_RECT,
            "ellipse": cv.MORPH_ELLIPSE,
            "cross": cv.MORPH_CROSS,
        }
        return cv.getStructuringElement(cv_shapes[element_shape], (width, height))

    def erode(
        self,
        mask: np.ndarray,
        element: np.ndarray,
        *,
        anchor: tuple[int, int] | None = None,
        iterations: int = 1,
        border: BorderPolicy = "constant",
    ) -> np.ndarray:
        """Erode primeiro plano binário segundo elemento, âncora e borda.

        ``iterations`` repete a erosão. ``anchor`` usa coordenadas ``(x, y)``
        dentro do elemento; ``None`` seleciona seu centro.
        """
        validated_mask = self._validate_mask(mask)
        validated_element = self._validate_element(element)
        validated_anchor = self._validate_anchor(anchor, validated_element)
        validated_iterations = self._validate_iterations(iterations)
        border_type = self._validate_border(border)
        return cv.erode(
            validated_mask,
            validated_element,
            anchor=validated_anchor,
            iterations=validated_iterations,
            borderType=border_type,
            borderValue=0,
        )

    def dilate(
        self,
        mask: np.ndarray,
        element: np.ndarray,
        *,
        anchor: tuple[int, int] | None = None,
        iterations: int = 1,
        border: BorderPolicy = "constant",
    ) -> np.ndarray:
        """Dilata primeiro plano binário segundo elemento, âncora e borda.

        ``iterations`` repete a dilatação. ``anchor`` usa coordenadas ``(x, y)``
        dentro do elemento; ``None`` seleciona seu centro.
        """
        validated_mask = self._validate_mask(mask)
        validated_element = self._validate_element(element)
        validated_anchor = self._validate_anchor(anchor, validated_element)
        validated_iterations = self._validate_iterations(iterations)
        border_type = self._validate_border(border)
        return cv.dilate(
            validated_mask,
            validated_element,
            anchor=validated_anchor,
            iterations=validated_iterations,
            borderType=border_type,
            borderValue=0,
        )

    def opening(
        self,
        mask: np.ndarray,
        element: np.ndarray,
        *,
        anchor: tuple[int, int] | None = None,
        iterations: int = 1,
        border: BorderPolicy = "constant",
    ) -> np.ndarray:
        """Aplica erosão seguida de dilatação à máscara binária.

        A abertura tende a remover objetos menores que o elemento. Os demais
        parâmetros têm a mesma semântica de :meth:`erode` e :meth:`dilate`.
        """
        eroded = self.erode(
            mask,
            element,
            anchor=anchor,
            iterations=iterations,
            border=border,
        )
        return self.dilate(
            eroded,
            element,
            anchor=anchor,
            iterations=iterations,
            border=border,
        )

    def closing(
        self,
        mask: np.ndarray,
        element: np.ndarray,
        *,
        anchor: tuple[int, int] | None = None,
        iterations: int = 1,
        border: BorderPolicy = "constant",
    ) -> np.ndarray:
        """Aplica dilatação seguida de erosão à máscara binária.

        O fechamento tende a preencher lacunas menores que o elemento. Os
        demais parâmetros têm a mesma semântica de :meth:`dilate` e
        :meth:`erode`.
        """
        dilated = self.dilate(
            mask,
            element,
            anchor=anchor,
            iterations=iterations,
            border=border,
        )
        return self.erode(
            dilated,
            element,
            anchor=anchor,
            iterations=iterations,
            border=border,
        )

    def gradient(
        self,
        mask: np.ndarray,
        element: np.ndarray,
        *,
        anchor: tuple[int, int] | None = None,
        iterations: int = 1,
        border: BorderPolicy = "constant",
    ) -> np.ndarray:
        """Calcula o gradiente binário: dilatação menos erosão.

        A saída marca com ``255`` os pixels que pertencem à dilatação e não à
        erosão, preservando o shape e dtype da máscara de entrada.
        """
        dilated = self.dilate(
            mask,
            element,
            anchor=anchor,
            iterations=iterations,
            border=border,
        )
        eroded = self.erode(
            mask,
            element,
            anchor=anchor,
            iterations=iterations,
            border=border,
        )
        return cv.subtract(dilated, eroded)

    @staticmethod
    def _validate_mask(mask: np.ndarray) -> np.ndarray:
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask deve ser um array NumPy.")
        if mask.ndim != 2 or mask.size == 0:
            raise ValueError("mask deve ser uma máscara binária 2D não vazia.")
        if mask.dtype != np.dtype(np.uint8):
            raise TypeError("mask deve possuir dtype uint8.")
        if not np.all((mask == 0) | (mask == 255)):
            raise ValueError("mask deve conter somente os valores binários 0 e 255.")
        return mask

    @staticmethod
    def _validate_element(element: np.ndarray) -> np.ndarray:
        if not isinstance(element, np.ndarray):
            raise TypeError("element deve ser um array NumPy.")
        if element.ndim != 2 or element.size == 0:
            raise ValueError("element deve ser um array 2D não vazio.")
        if element.dtype != np.dtype(np.uint8):
            raise TypeError("element deve possuir dtype uint8.")
        if any(dimension % 2 == 0 for dimension in element.shape):
            raise ValueError("As dimensões de element devem ser ímpares.")
        if not np.all((element == 0) | (element == 1)):
            raise ValueError("element deve conter somente os valores 0 e 1.")
        if not np.any(element):
            raise ValueError("element deve possuir ao menos uma posição ativa.")
        return element

    @staticmethod
    def _validate_element_shape(
        shape: StructuringElementShape,
    ) -> StructuringElementShape:
        if not isinstance(shape, str) or shape not in _STRUCTURING_ELEMENT_SHAPES:
            raise ValueError("shape deve ser 'rectangle', 'ellipse' ou 'cross'.")
        return shape

    @staticmethod
    def _validate_size(size: tuple[int, int]) -> tuple[int, int]:
        if not isinstance(size, tuple) or len(size) != 2:
            raise TypeError("size deve ser uma tupla (width, height).")
        width, height = size
        for value, name in ((width, "width"), (height, "height")):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} deve ser um número inteiro.")
            if value <= 0:
                raise ValueError(f"{name} deve ser maior que zero.")
            if value % 2 == 0:
                raise ValueError(f"{name} deve ser ímpar.")
        return int(width), int(height)

    @staticmethod
    def _validate_anchor(
        anchor: tuple[int, int] | None,
        element: np.ndarray,
    ) -> tuple[int, int]:
        if anchor is None:
            return (-1, -1)
        if not isinstance(anchor, tuple) or len(anchor) != 2:
            raise TypeError("anchor deve ser uma tupla (x, y) ou None.")
        x, y = anchor
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in anchor
        ):
            raise TypeError("anchor deve conter números inteiros.")
        if not 0 <= x < element.shape[1] or not 0 <= y < element.shape[0]:
            raise ValueError("anchor deve estar dentro dos limites de element.")
        return int(x), int(y)

    @staticmethod
    def _validate_iterations(iterations: int) -> int:
        if isinstance(iterations, (bool, np.bool_)) or not isinstance(
            iterations, Integral
        ):
            raise TypeError("iterations deve ser um número inteiro.")
        if iterations <= 0:
            raise ValueError("iterations deve ser maior que zero.")
        return int(iterations)

    @staticmethod
    def _validate_border(border: BorderPolicy) -> int:
        if not isinstance(border, str) or border not in _BORDER_POLICIES:
            raise ValueError("border deve ser 'constant' ou 'replicate'.")
        return cv.BORDER_CONSTANT if border == "constant" else cv.BORDER_REPLICATE
