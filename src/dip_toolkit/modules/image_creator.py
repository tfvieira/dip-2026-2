"""Criação determinística de imagens sintéticas representadas por arrays NumPy."""

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Literal, TypeAlias

import numpy as np
from numpy import typing as npt

ImageShape: TypeAlias = tuple[int, int] | tuple[int, int, int]
ImageShapeInput: TypeAlias = Sequence[int]
Distribution: TypeAlias = Literal["uniform", "normal", "rayleigh"]
Seed: TypeAlias = int | np.integer


class ImageCreator:
    """Cria imagens constantes, aleatórias e com ruído impulsivo.

    Os métodos aceitam imagens em escala de cinza, com shape ``(altura, largura)``,
    ou imagens multicanais, com shape ``(altura, largura, canais)``. Dtypes
    inteiros usam o intervalo não negativo ``[0, max(dtype)]`` nas gerações
    aleatórias; dtypes de ponto flutuante usam ``[0.0, 1.0]``.

    Parameters
    ----------
    seed:
        Semente não negativa usada pelo gerador interno. Não pode ser combinada
        com ``rng``.
    rng:
        Gerador NumPy usado internamente. Seu estado avança a cada geração.
    """

    def __init__(
        self,
        *,
        seed: Seed | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._rng = self._resolve_rng(seed=seed, rng=rng)

    def create_filled_image(
        self,
        shape: ImageShapeInput,
        value: Real = 0,
        dtype: npt.DTypeLike = np.uint8,
    ) -> np.ndarray:
        """Cria uma imagem preenchida por um valor escalar.

        Para dtypes inteiros, ``value`` deve ser inteiro e estar no intervalo
        completo do dtype. Para ponto flutuante, o valor deve ser finito e
        representável no dtype.

        Parameters
        ----------
        shape:
            Sequência, como tuple ou list, com duas ou três dimensões positivas.
        value:
            Valor real usado para preencher todos os pixels.
        dtype:
            Dtype NumPy inteiro ou de ponto flutuante.

        Returns
        -------
        numpy.ndarray
            Novo array com o shape, dtype e valor solicitados.
        """

        validated_shape = self._validate_shape(shape)
        validated_dtype = self._validate_dtype(dtype)
        validated_value = self._validate_fill_value(value, validated_dtype)
        return np.full(validated_shape, validated_value, dtype=validated_dtype)

    def create_random_image(
        self,
        shape: ImageShapeInput,
        distribution: Distribution = "uniform",
        dtype: npt.DTypeLike = np.uint8,
        *,
        low: Real | None = None,
        high: Real | None = None,
        mean: Real | None = None,
        std: Real | None = None,
        scale: Real | None = None,
        seed: Seed | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Cria uma imagem aleatória uniforme, normal ou Rayleigh.

        A distribuição uniforme usa o intervalo semiaberto ``[low, high)``.
        Para inteiros, os limites devem ser números inteiros e o padrão é
        ``[0, max(dtype) + 1)``, incluindo assim todo o domínio não negativo.
        Para floats, o padrão é ``[0.0, 1.0)``.

        Nas distribuições normal e Rayleigh, as amostras são limitadas ao
        domínio da imagem: ``[0, max(dtype)]`` para inteiros e ``[0.0, 1.0]``
        para floats. Amostras inteiras são arredondadas para o inteiro mais
        próximo antes da conversão.

        Os parâmetros padrão preservam escalas úteis em cada dtype. A normal usa
        o ponto médio do domínio; seu desvio padrão é ``20`` em ``uint8`` e
        ``0.1`` em floats. A Rayleigh usa escala ``50`` em ``uint8`` e ``0.25``
        em floats, com ajuste proporcional para outros dtypes inteiros.

        Parameters
        ----------
        shape:
            Sequência, como tuple ou list, com duas ou três dimensões positivas.
        distribution:
            ``"uniform"``, ``"normal"`` ou ``"rayleigh"``.
        dtype:
            Dtype NumPy inteiro ou de ponto flutuante.
        low, high:
            Limites da distribuição uniforme. Use somente com ``"uniform"``.
        mean, std:
            Média e desvio padrão positivo. Use somente com ``"normal"``.
        scale:
            Escala positiva. Use somente com ``"rayleigh"``.
        seed:
            Semente não negativa para esta chamada. Não pode ser combinada com
            ``rng``.
        rng:
            Gerador NumPy para esta chamada. Seu estado será avançado.

        Returns
        -------
        numpy.ndarray
            Nova imagem no dtype e no domínio documentados.
        """

        validated_shape = self._validate_shape(shape)
        validated_dtype = self._validate_dtype(dtype)
        generator = self._rng_for_call(seed=seed, rng=rng)

        if distribution == "uniform":
            self._reject_parameters(
                distribution,
                mean=mean,
                std=std,
                scale=scale,
            )
            return self._create_uniform_image(
                validated_shape,
                validated_dtype,
                generator,
                low=low,
                high=high,
            )

        if distribution == "normal":
            self._reject_parameters(
                distribution,
                low=low,
                high=high,
                scale=scale,
            )
            upper = self._image_upper_bound(validated_dtype)
            default_std = (
                upper * (20 / 255)
                if np.issubdtype(validated_dtype, np.integer)
                else 0.1
            )
            validated_mean = self._validate_parameter(
                "mean",
                upper / 2 if mean is None else mean,
                minimum=0.0,
                maximum=upper,
            )
            validated_std = self._validate_parameter(
                "std",
                default_std if std is None else std,
                minimum=0.0,
                minimum_inclusive=False,
            )
            samples = generator.normal(
                validated_mean,
                validated_std,
                validated_shape,
            )
            return self._cast_random_samples(samples, validated_dtype)

        if distribution == "rayleigh":
            self._reject_parameters(
                distribution,
                low=low,
                high=high,
                mean=mean,
                std=std,
            )
            upper = self._image_upper_bound(validated_dtype)
            default_scale = (
                upper * (50 / 255)
                if np.issubdtype(validated_dtype, np.integer)
                else 0.25
            )
            validated_scale = self._validate_parameter(
                "scale",
                default_scale if scale is None else scale,
                minimum=0.0,
                minimum_inclusive=False,
            )
            samples = generator.rayleigh(validated_scale, validated_shape)
            return self._cast_random_samples(samples, validated_dtype)

        if not isinstance(distribution, str):
            raise TypeError("distribution deve ser uma string.")
        raise ValueError(
            "Distribuição inválida. Use 'uniform', 'normal' ou 'rayleigh'."
        )

    def change_image_dtype(
        self,
        image: np.ndarray,
        dtype: npt.DTypeLike,
    ) -> np.ndarray:
        """Converte o dtype de uma imagem sem alterar o array recebido.

        Conversões para inteiros seguem a regra do NumPy e truncam a parte
        fracionária em direção a zero. Valores não finitos ou fora do intervalo
        representável pelo dtype de destino são rejeitados, evitando overflow
        silencioso. Um novo array é retornado mesmo se o dtype não mudar.

        Parameters
        ----------
        image:
            Array NumPy 2D ou 3D, não vazio e com dtype real.
        dtype:
            Dtype NumPy inteiro ou de ponto flutuante para o resultado.

        Returns
        -------
        numpy.ndarray
            Cópia convertida da imagem.
        """

        self._validate_image(image)
        validated_dtype = self._validate_dtype(dtype)

        if np.issubdtype(image.dtype, np.floating) and not np.all(np.isfinite(image)):
            raise ValueError("image deve conter somente valores finitos.")

        limits = (
            np.iinfo(validated_dtype)
            if np.issubdtype(validated_dtype, np.integer)
            else np.finfo(validated_dtype)
        )
        if np.any(image < limits.min) or np.any(image > limits.max):
            raise ValueError(
                f"image contém valores fora do intervalo de {validated_dtype}."
            )

        return image.astype(validated_dtype, copy=True)

    def create_zeros_image(
        self,
        shape: ImageShapeInput,
        dtype: npt.DTypeLike = np.uint8,
    ) -> np.ndarray:
        """Cria uma imagem preenchida pelo valor numérico zero.

        Parameters
        ----------
        shape:
            Sequência, como tuple ou list, com duas ou três dimensões positivas.
        dtype:
            Dtype NumPy inteiro ou de ponto flutuante.

        Returns
        -------
        numpy.ndarray
            Novo array contendo zero em todos os pixels.
        """

        validated_shape = self._validate_shape(shape)
        validated_dtype = self._validate_dtype(dtype)
        return np.zeros(validated_shape, dtype=validated_dtype)

    def create_ones_image(
        self,
        shape: ImageShapeInput,
        dtype: npt.DTypeLike = np.uint8,
    ) -> np.ndarray:
        """Cria uma imagem de uns no dtype solicitado.

        Para dtypes inteiros, todos os pixels recebem ``max(dtype)``. Assim, uma
        imagem ``uint8`` contém ``255`` e é visualizada como branca. Para dtypes
        de ponto flutuante, todos os pixels recebem ``1.0``.

        Parameters
        ----------
        shape:
            Sequência, como tuple ou list, com duas ou três dimensões positivas.
        dtype:
            Dtype NumPy inteiro ou de ponto flutuante.

        Returns
        -------
        numpy.ndarray
            Novo array preenchido pelo nível de um do dtype.
        """

        validated_shape = self._validate_shape(shape)
        validated_dtype = self._validate_dtype(dtype)
        value: int | float
        if np.issubdtype(validated_dtype, np.integer):
            value = int(np.iinfo(validated_dtype).max)
        else:
            value = 1.0
        return np.full(validated_shape, value, dtype=validated_dtype)

    def create_salt_and_pepper_noise(
        self,
        height: int = 100,
        width: int = 100,
        salt_prob: Real = 0.05,
        pepper_prob: Real = 0.05,
        *,
        dtype: npt.DTypeLike = np.float64,
        seed: Seed | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Cria uma imagem 2D com ruído impulsivo sal-e-pimenta.

        Cada pixel pertence a uma única classe. A probabilidade de pimenta é
        ``pepper_prob``, a de sal é ``salt_prob`` e a de fundo é
        ``1 - salt_prob - pepper_prob``. As probabilidades devem pertencer a
        ``[0, 1]`` e sua soma não pode ultrapassar ``1``.

        Para floats, pimenta, fundo e sal valem ``-1.0``, ``0.5`` e ``1.0``.
        Para inteiros, valem ``0``, ``max(dtype) // 2`` e ``max(dtype)``.

        Parameters
        ----------
        height, width:
            Dimensões positivas da imagem.
        salt_prob, pepper_prob:
            Probabilidades de impulsos de sal e pimenta.
        dtype:
            Dtype NumPy inteiro ou de ponto flutuante.
        seed:
            Semente não negativa para esta chamada. Não pode ser combinada com
            ``rng``.
        rng:
            Gerador NumPy para esta chamada. Seu estado será avançado.

        Returns
        -------
        numpy.ndarray
            Nova imagem 2D no dtype solicitado.
        """

        validated_shape = self._validate_shape((height, width))
        validated_dtype = self._validate_dtype(dtype)
        validated_salt = self._validate_probability("salt_prob", salt_prob)
        validated_pepper = self._validate_probability("pepper_prob", pepper_prob)
        probability_sum = validated_salt + validated_pepper
        if probability_sum > 1.0 and not np.isclose(
            probability_sum,
            1.0,
            rtol=1e-7,
            atol=0.0,
        ):
            raise ValueError("salt_prob + pepper_prob não pode ultrapassar 1.")

        upper = self._image_upper_bound(validated_dtype)
        if np.issubdtype(validated_dtype, np.integer):
            background: Real = int(upper) // 2
            pepper: Real = 0
        else:
            background = 0.5
            pepper = -1.0

        image = np.full(validated_shape, background, dtype=validated_dtype)
        random_values = self._rng_for_call(seed=seed, rng=rng).random(validated_shape)
        pepper_mask = random_values < validated_pepper
        salt_mask = (random_values >= validated_pepper) & (
            random_values < validated_pepper + validated_salt
        )
        image[pepper_mask] = pepper
        image[salt_mask] = upper
        return image

    @staticmethod
    def _validate_shape(shape: ImageShapeInput) -> ImageShape:
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
            raise TypeError("shape deve ser uma sequência, como tuple ou list.")
        if len(shape) not in {2, 3}:
            raise ValueError("shape deve ter duas ou três dimensões.")
        if any(
            isinstance(dimension, (bool, np.bool_))
            or not isinstance(dimension, Integral)
            for dimension in shape
        ):
            raise TypeError("As dimensões de shape devem ser números inteiros.")
        if any(dimension <= 0 for dimension in shape):
            raise ValueError("As dimensões de shape devem ser positivas.")
        return tuple(int(dimension) for dimension in shape)  # type: ignore[return-value]

    @staticmethod
    def _validate_dtype(dtype: npt.DTypeLike) -> np.dtype:
        if dtype is None:
            raise TypeError("dtype deve ser um dtype NumPy inteiro ou float.")
        try:
            validated_dtype = np.dtype(dtype)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "dtype deve ser um dtype NumPy inteiro ou float."
            ) from error

        if not (
            np.issubdtype(validated_dtype, np.integer)
            or np.issubdtype(validated_dtype, np.floating)
        ):
            raise TypeError("dtype deve ser um dtype NumPy inteiro ou float.")
        return validated_dtype

    @classmethod
    def _validate_image(cls, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image deve ser um array NumPy.")
        cls._validate_shape(image.shape)
        cls._validate_dtype(image.dtype)

    @staticmethod
    def _validate_fill_value(value: Real, dtype: np.dtype) -> int | float:
        validated_value = ImageCreator._validate_real("value", value)
        limits = (
            np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)
        )
        if validated_value < limits.min or validated_value > limits.max:
            raise ValueError(f"value está fora do intervalo de {dtype}.")
        if np.issubdtype(dtype, np.integer):
            if not validated_value.is_integer():
                raise ValueError("value deve ser inteiro para um dtype inteiro.")
            return int(validated_value)
        return validated_value

    @staticmethod
    def _validate_real(name: str, value: Real) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} deve ser um número real.")
        try:
            validated_value = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError(f"{name} deve ser um número real finito.") from error
        if not np.isfinite(validated_value):
            raise ValueError(f"{name} deve ser um número real finito.")
        return validated_value

    @classmethod
    def _validate_parameter(
        cls,
        name: str,
        value: Real,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        minimum_inclusive: bool = True,
    ) -> float:
        validated_value = cls._validate_real(name, value)
        if minimum is not None:
            invalid_minimum = (
                validated_value < minimum
                if minimum_inclusive
                else validated_value <= minimum
            )
            if invalid_minimum:
                comparator = ">=" if minimum_inclusive else ">"
                raise ValueError(f"{name} deve ser {comparator} {minimum}.")
        if maximum is not None and validated_value > maximum:
            raise ValueError(f"{name} deve ser <= {maximum}.")
        return validated_value

    @classmethod
    def _validate_probability(cls, name: str, value: Real) -> float:
        return cls._validate_parameter(
            name,
            value,
            minimum=0.0,
            maximum=1.0,
        )

    @staticmethod
    def _image_upper_bound(dtype: np.dtype) -> int | float:
        if np.issubdtype(dtype, np.integer):
            return int(np.iinfo(dtype).max)
        return 1.0

    @classmethod
    def _create_uniform_image(
        cls,
        shape: ImageShape,
        dtype: np.dtype,
        rng: np.random.Generator,
        *,
        low: Real | None,
        high: Real | None,
    ) -> np.ndarray:
        if np.issubdtype(dtype, np.integer):
            upper = np.iinfo(dtype).max
            validated_low = cls._validate_parameter(
                "low",
                0 if low is None else low,
                minimum=0.0,
                maximum=float(upper),
            )
            validated_high = cls._validate_parameter(
                "high",
                int(upper) + 1 if high is None else high,
                minimum=0.0,
                maximum=float(int(upper) + 1),
            )
            if not validated_low.is_integer() or not validated_high.is_integer():
                raise ValueError("low e high devem ser inteiros para um dtype inteiro.")
            if validated_low >= validated_high:
                raise ValueError("low deve ser menor que high.")
            return rng.integers(
                int(validated_low),
                int(validated_high),
                size=shape,
                dtype=dtype,
            )

        validated_low = cls._validate_parameter(
            "low",
            0.0 if low is None else low,
            minimum=0.0,
            maximum=1.0,
        )
        validated_high = cls._validate_parameter(
            "high",
            1.0 if high is None else high,
            minimum=0.0,
            maximum=1.0,
        )
        if validated_low >= validated_high:
            raise ValueError("low deve ser menor que high.")
        return rng.uniform(validated_low, validated_high, shape).astype(dtype)

    @classmethod
    def _cast_random_samples(
        cls,
        samples: np.ndarray,
        dtype: np.dtype,
    ) -> np.ndarray:
        upper = cls._image_upper_bound(dtype)
        clipped = np.clip(samples, 0.0, upper)
        if np.issubdtype(dtype, np.integer):
            clipped = np.rint(clipped)
        return clipped.astype(dtype)

    @staticmethod
    def _reject_parameters(distribution: str, **parameters: Real | None) -> None:
        supplied = [name for name, value in parameters.items() if value is not None]
        if supplied:
            names = ", ".join(supplied)
            raise ValueError(
                f"Parâmetro(s) {names} não se aplica(m) à distribuição "
                f"{distribution!r}."
            )

    def _rng_for_call(
        self,
        *,
        seed: Seed | None,
        rng: np.random.Generator | None,
    ) -> np.random.Generator:
        if seed is None and rng is None:
            return self._rng
        return self._resolve_rng(seed=seed, rng=rng)

    @staticmethod
    def _resolve_rng(
        *,
        seed: Seed | None,
        rng: np.random.Generator | None,
    ) -> np.random.Generator:
        if seed is not None and rng is not None:
            raise ValueError("Use seed ou rng, não ambos.")
        if rng is not None:
            if not isinstance(rng, np.random.Generator):
                raise TypeError("rng deve ser uma instância de numpy.random.Generator.")
            return rng
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
                raise TypeError("seed deve ser um número inteiro não negativo.")
            if seed < 0:
                raise ValueError("seed deve ser um número inteiro não negativo.")
            seed = int(seed)
        return np.random.default_rng(seed)
