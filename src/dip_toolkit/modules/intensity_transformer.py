from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import cv2 as cv
import numpy as np

ValueRange = tuple[Real, Real]
ControlPoint = tuple[Real, Real]


class IntensityTransformer:
    """Aplica transformações ponto a ponto e equalização em imagens.

    As transformações ponto a ponto aceitam imagens grayscale 2D ou coloridas
    3D com três canais e preservam shape e dtype. Para inteiros, a faixa padrão
    é a faixa completa de numpy.iinfo(dtype). Para floats, valores em [0.0, 1.0]
    são aceitos automaticamente; outras faixas precisam ser informadas por
    value_range.
    """

    def negative(
        self,
        image: np.ndarray,
        *,
        value_range: ValueRange | None = None,
    ) -> np.ndarray:
        """Calcula o negativo lower + upper - image.

        Args:
            image: Imagem 2D ou imagem 3D com exatamente três canais.
            value_range: Faixa inclusiva (mínimo, máximo) da transformação.

        Returns:
            Novo array na mesma faixa, com shape e dtype preservados.
        """

        lower, upper = self._prepare(image, value_range)
        transformed = lower + upper - image.astype(np.float64)
        return self._restore_dtype(transformed, image, lower, upper)

    def log_transform(
        self,
        image: np.ndarray,
        *,
        log_gain: Real = 1.0,
        value_range: ValueRange | None = None,
    ) -> np.ndarray:
        """Aplica uma transformação logarítmica normalizada.

        A fórmula é log1p(log_gain * x) / log1p(log_gain), em que x é
        a entrada normalizada em [0, 1]. Os extremos permanecem nos extremos da
        faixa original. Inteiros são arredondados antes da conversão ao dtype.

        Args:
            image: Imagem 2D ou imagem 3D com exatamente três canais.
            log_gain: Ganho real, finito e estritamente positivo.
            value_range: Faixa inclusiva (mínimo, máximo) da transformação.

        Returns:
            Novo array transformado, com o mesmo shape e dtype da entrada.
        """

        validated_gain = self._validate_positive_real(log_gain, "log_gain")
        lower, upper = self._prepare(image, value_range)
        normalized = self._normalize(image, lower, upper)
        transformed = np.log1p(validated_gain * normalized) / np.log1p(validated_gain)
        remapped = lower + transformed * (upper - lower)
        return self._restore_dtype(remapped, image, lower, upper)

    def gamma_transform(
        self,
        image: np.ndarray,
        *,
        gamma: Real,
        gain: Real = 1.0,
        value_range: ValueRange | None = None,
    ) -> np.ndarray:
        """Aplica a transformação gama normalizada.

        Calcula gain * normalized**gamma e limita explicitamente o resultado
        normalizado a [0, 1]. Ganhos maiores que um podem saturar valores no
        limite superior. O resultado é remapeado para a faixa original.

        Args:
            image: Imagem 2D ou imagem 3D com exatamente três canais.
            gamma: Expoente real, finito e estritamente positivo.
            gain: Ganho real, finito e estritamente positivo.
            value_range: Faixa inclusiva (mínimo, máximo) da transformação.

        Returns:
            Novo array transformado, com o mesmo shape e dtype da entrada.
        """

        validated_gamma = self._validate_positive_real(gamma, "gamma")
        validated_gain = self._validate_positive_real(gain, "gain")
        lower, upper = self._prepare(image, value_range)
        normalized = self._normalize(image, lower, upper)
        transformed = validated_gain * np.power(normalized, validated_gamma)
        transformed = np.clip(transformed, 0.0, 1.0)
        remapped = lower + transformed * (upper - lower)
        return self._restore_dtype(remapped, image, lower, upper)

    def piecewise_linear(
        self,
        image: np.ndarray,
        control_points: Sequence[ControlPoint],
        *,
        value_range: ValueRange | None = None,
    ) -> np.ndarray:
        """Interpola uma transformação linear por pontos de controle.

        As coordenadas usam a faixa real da imagem. Abaixo do primeiro ponto, a
        saída é constante e igual à saída desse ponto; acima do último, é igual
        à saída do último ponto.

        Args:
            image: Imagem 2D ou imagem 3D com exatamente três canais.
            control_points: Ao menos dois pares (entrada, saída), com entradas
                estritamente crescentes.
            value_range: Faixa inclusiva (mínimo, máximo) da transformação.

        Returns:
            Novo array interpolado, com o mesmo shape e dtype da entrada.
        """

        lower, upper = self._prepare(image, value_range)
        inputs, outputs = self._validate_control_points(
            control_points,
            lower,
            upper,
        )
        transformed = np.interp(
            image.astype(np.float64),
            inputs,
            outputs,
            left=outputs[0],
            right=outputs[-1],
        )
        return self._restore_dtype(transformed, image, lower, upper)

    def equalize_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Equaliza o histograma de uma imagem grayscale uint8.

        Imagens constantes retornam uma cópia equivalente. Imagens coloridas,
        vazias ou com outro dtype são rejeitadas. A entrada nunca é modificada.

        Args:
            image: Imagem grayscale 2D com dtype uint8.

        Returns:
            Nova imagem equalizada, com shape e dtype preservados.
        """

        self._validate_image(image)
        if image.ndim != 2:
            raise ValueError("equalize_grayscale aceita somente imagens grayscale 2D.")
        if image.dtype != np.dtype(np.uint8):
            raise TypeError("equalize_grayscale aceita somente imagens uint8.")
        if np.min(image) == np.max(image):
            return image.copy()
        return cv.equalizeHist(image)

    def _prepare(
        self,
        image: np.ndarray,
        value_range: ValueRange | None,
    ) -> tuple[float, float]:
        self._validate_image(image)
        lower, upper = self._resolve_value_range(image, value_range)
        if np.any(image < lower) or np.any(image > upper):
            raise ValueError("image contém valores fora de value_range.")
        return lower, upper

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
        if np.issubdtype(image.dtype, np.bool_):
            raise TypeError("image não aceita dtype booleano.")
        if not (
            np.issubdtype(image.dtype, np.integer)
            or np.issubdtype(image.dtype, np.floating)
        ):
            raise TypeError("image deve usar um dtype NumPy inteiro ou float.")
        if np.issubdtype(image.dtype, np.floating) and not np.all(np.isfinite(image)):
            raise ValueError("image deve conter somente valores finitos.")

    @staticmethod
    def _resolve_value_range(
        image: np.ndarray,
        value_range: ValueRange | None,
    ) -> tuple[float, float]:
        if value_range is None:
            if np.issubdtype(image.dtype, np.integer):
                limits = np.iinfo(image.dtype)
                return float(limits.min), float(limits.max)
            if np.any(image < 0.0) or np.any(image > 1.0):
                raise ValueError(
                    "Imagens float fora de [0.0, 1.0] devem informar "
                    "value_range explicitamente."
                )
            return 0.0, 1.0

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
        return lower, upper

    @staticmethod
    def _validate_positive_real(value: Real, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} deve ser um número real.")
        validated = float(value)
        if not np.isfinite(validated):
            raise ValueError(f"{name} deve ser finito.")
        if validated <= 0.0:
            raise ValueError(f"{name} deve ser maior que zero.")
        return validated

    @staticmethod
    def _normalize(
        image: np.ndarray,
        lower: float,
        upper: float,
    ) -> np.ndarray:
        return (image.astype(np.float64) - lower) / (upper - lower)

    @staticmethod
    def _restore_dtype(
        transformed: np.ndarray,
        image: np.ndarray,
        lower: float,
        upper: float,
    ) -> np.ndarray:
        if not np.all(np.isfinite(transformed)):
            raise ValueError("A transformação produziu valores não finitos.")

        scale = max(1.0, abs(lower), abs(upper))
        tolerance = np.finfo(np.float64).eps * scale * 16
        if np.any(transformed < lower - tolerance) or np.any(
            transformed > upper + tolerance
        ):
            raise ValueError("A transformação produziu valores fora de value_range.")
        bounded = np.clip(transformed, lower, upper)

        if np.issubdtype(image.dtype, np.floating):
            return bounded.astype(image.dtype, copy=True)

        rounded = np.rint(bounded)
        limits = np.iinfo(image.dtype)
        lower_limit = float(limits.min)
        upper_limit = float(limits.max)
        if np.any(rounded < lower_limit) or np.any(rounded > upper_limit):
            raise OverflowError("A transformação excede a faixa do dtype de saída.")

        result = np.empty(image.shape, dtype=image.dtype)
        lower_mask = rounded <= lower_limit
        upper_mask = rounded >= upper_limit
        middle_mask = ~(lower_mask | upper_mask)
        result[lower_mask] = limits.min
        result[upper_mask] = limits.max
        result[middle_mask] = rounded[middle_mask].astype(image.dtype)
        return result

    @staticmethod
    def _validate_control_points(
        control_points: Sequence[ControlPoint],
        lower: float,
        upper: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(control_points, (str, bytes)) or not isinstance(
            control_points, Sequence
        ):
            raise TypeError("control_points deve ser uma sequência de pares.")
        if len(control_points) < 2:
            raise ValueError("control_points deve possuir pelo menos dois pontos.")

        validated: list[tuple[float, float]] = []
        for point in control_points:
            if isinstance(point, (str, bytes)) or not isinstance(point, Sequence):
                raise TypeError("Cada ponto de controle deve ser um par.")
            if len(point) != 2:
                raise ValueError("Cada ponto de controle deve possuir dois valores.")
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                for value in point
            ):
                raise TypeError("Pontos de controle devem conter números reais.")
            point_input, point_output = (float(value) for value in point)
            if not np.isfinite(point_input) or not np.isfinite(point_output):
                raise ValueError("Pontos de controle devem conter valores finitos.")
            if not (lower <= point_input <= upper and lower <= point_output <= upper):
                raise ValueError("Pontos de controle devem pertencer a value_range.")
            validated.append((point_input, point_output))

        inputs = np.asarray([point[0] for point in validated], dtype=np.float64)
        if np.any(np.diff(inputs) <= 0):
            raise ValueError(
                "As entradas dos pontos de controle devem ser estritamente crescentes."
            )
        outputs = np.asarray([point[1] for point in validated], dtype=np.float64)
        return inputs, outputs
