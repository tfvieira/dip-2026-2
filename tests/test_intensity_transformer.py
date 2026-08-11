import numpy as np
import pytest

from dip_toolkit.modules.intensity_transformer import IntensityTransformer


def test_negative_uint8_has_expected_values_and_preserves_input() -> None:
    image = np.array([[0, 64, 255]], dtype=np.uint8)
    original = image.copy()

    result = IntensityTransformer().negative(image)

    assert np.array_equal(result, np.array([[255, 191, 0]], dtype=np.uint8))
    assert np.array_equal(image, original)
    assert result is not image


def test_negative_float_unit_range() -> None:
    image = np.array([[0.0, 0.25, 1.0]], dtype=np.float32)

    result = IntensityTransformer().negative(image)

    assert result.dtype == image.dtype
    assert np.allclose(result, [[1.0, 0.75, 0.0]])


def test_negative_float_explicit_signed_range() -> None:
    image = np.array([[-1.0, 0.0, 1.0]], dtype=np.float64)

    result = IntensityTransformer().negative(
        image,
        value_range=(-1.0, 1.0),
    )

    assert np.array_equal(result, np.array([[1.0, 0.0, -1.0]]))


def test_point_transform_preserves_color_shape_channels_and_dtype() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    result = IntensityTransformer().negative(image)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert not np.shares_memory(result, image)


def test_log_transform_preserves_extremes_and_is_monotonic() -> None:
    image = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float64)

    result = IntensityTransformer().log_transform(image, log_gain=1.0)

    assert result[0, 0] == pytest.approx(0.0)
    assert result[0, -1] == pytest.approx(1.0)
    assert np.all(np.diff(result.ravel()) >= 0)
    assert result[0, 2] == pytest.approx(np.log1p(0.5) / np.log(2.0))


@pytest.mark.parametrize(
    ("log_gain", "exception"),
    [
        (0.0, ValueError),
        (-1.0, ValueError),
        (np.inf, ValueError),
        (np.nan, ValueError),
        ("1", TypeError),
        (True, TypeError),
    ],
)
def test_log_transform_rejects_invalid_gain(
    log_gain: object,
    exception: type[Exception],
) -> None:
    image = np.array([[0.0, 1.0]], dtype=np.float32)

    with pytest.raises(exception):
        IntensityTransformer().log_transform(
            image,
            log_gain=log_gain,  # type: ignore[arg-type]
        )


def test_gamma_identity() -> None:
    image = np.array([[0.0, 0.25, 0.5, 1.0]], dtype=np.float32)

    result = IntensityTransformer().gamma_transform(
        image,
        gamma=1.0,
        gain=1.0,
    )

    assert np.array_equal(result, image)
    assert result is not image


def test_gamma_squared_ramp() -> None:
    image = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)

    result = IntensityTransformer().gamma_transform(image, gamma=2.0)

    assert np.allclose(result, [[0.0, 0.25, 1.0]])


def test_gamma_gain_clips_at_upper_limit() -> None:
    image = np.array([[0.25, 0.75]], dtype=np.float32)

    result = IntensityTransformer().gamma_transform(
        image,
        gamma=1.0,
        gain=2.0,
    )

    assert np.allclose(result, [[0.5, 1.0]])


@pytest.mark.parametrize("name", ["gamma", "gain"])
@pytest.mark.parametrize(
    ("value", "exception"),
    [
        (0.0, ValueError),
        (-1.0, ValueError),
        (np.inf, ValueError),
        (np.nan, ValueError),
        ("1", TypeError),
        (True, TypeError),
    ],
)
def test_gamma_transform_rejects_invalid_parameters(
    name: str,
    value: object,
    exception: type[Exception],
) -> None:
    image = np.array([[0.0, 1.0]], dtype=np.float32)
    arguments: dict[str, object] = {"gamma": 1.0, "gain": 1.0}
    arguments[name] = value

    with pytest.raises(exception):
        IntensityTransformer().gamma_transform(
            image,
            **arguments,  # type: ignore[arg-type]
        )


def test_piecewise_linear_interpolates_and_extends_endpoints() -> None:
    image = np.array([[0, 50, 100, 150, 200, 255]], dtype=np.uint8)

    result = IntensityTransformer().piecewise_linear(
        image,
        [(50, 20), (200, 240)],
    )

    assert np.array_equal(
        result,
        np.array([[20, 20, 93, 167, 240, 240]], dtype=np.uint8),
    )


@pytest.mark.parametrize(
    ("points", "exception"),
    [
        ([(0, 0)], ValueError),
        ([(0, 0), (0, 255)], ValueError),
        ([(100, 0), (50, 255)], ValueError),
        ([(0, 0), (256, 255)], ValueError),
        ([(0, -1), (255, 255)], ValueError),
        ([(0, 0), (np.nan, 255)], ValueError),
        ([(0, 0), (np.inf, 255)], ValueError),
        ([(0, 0, 0), (255, 255)], ValueError),
        ([(0, 0), ("255", 255)], TypeError),
    ],
)
def test_piecewise_linear_rejects_invalid_points(
    points: object,
    exception: type[Exception],
) -> None:
    image = np.array([[0, 255]], dtype=np.uint8)

    with pytest.raises(exception):
        IntensityTransformer().piecewise_linear(
            image,
            points,  # type: ignore[arg-type]
        )


def test_equalize_grayscale_bimodal_image() -> None:
    image = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    original = image.copy()

    result = IntensityTransformer().equalize_grayscale(image)

    assert np.array_equal(result, np.array([[0, 0], [255, 255]], dtype=np.uint8))
    assert np.array_equal(image, original)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_equalize_grayscale_applies_expected_cdf_lut() -> None:
    image = np.array(
        [
            [0, 0, 0, 1],
            [1, 1, 2, 2],
            [2, 3, 3, 3],
            [3, 3, 3, 3],
        ],
        dtype=np.uint8,
    )
    original = image.copy()
    expected = np.array(
        [
            [0, 0, 0, 59],
            [59, 59, 118, 118],
            [118, 255, 255, 255],
            [255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )

    result = IntensityTransformer().equalize_grayscale(image)

    assert np.array_equal(result, expected)
    assert np.array_equal(image, original)
    assert result is not image


def test_equalize_grayscale_constant_returns_equivalent_copy() -> None:
    image = np.full((2, 3), 42, dtype=np.uint8)

    result = IntensityTransformer().equalize_grayscale(image)

    assert np.array_equal(result, image)
    assert result is not image


@pytest.mark.parametrize(
    ("image", "exception"),
    [
        (np.zeros((2, 2, 3), dtype=np.uint8), ValueError),
        (np.zeros((2, 2), dtype=np.uint16), TypeError),
        (np.empty((0, 2), dtype=np.uint8), ValueError),
        (np.zeros(3, dtype=np.uint8), ValueError),
    ],
)
def test_equalize_grayscale_rejects_invalid_image(
    image: np.ndarray,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        IntensityTransformer().equalize_grayscale(image)


@pytest.mark.parametrize(
    ("image", "exception"),
    [
        (np.array([], dtype=np.uint8), ValueError),
        (np.ones(3, dtype=np.uint8), ValueError),
        (np.ones((2, 2, 2), dtype=np.uint8), ValueError),
        (np.ones((2, 2, 4), dtype=np.uint8), ValueError),
        (np.ones((2, 2), dtype=np.bool_), TypeError),
        (np.array([[np.nan]], dtype=np.float32), ValueError),
        (np.array([[np.inf]], dtype=np.float32), ValueError),
        (np.array([[-np.inf]], dtype=np.float32), ValueError),
    ],
)
def test_point_transform_rejects_invalid_images(
    image: np.ndarray,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        IntensityTransformer().negative(image)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_point_transform_rejects_unsupported_64_bit_integer_dtype(
    dtype: type[np.generic],
) -> None:
    image = np.array([[0, 1]], dtype=dtype)

    with pytest.raises(TypeError, match="dtype"):
        IntensityTransformer().negative(image)


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.int16, np.float32, np.float64],
)
def test_point_transform_accepts_supported_dtype(dtype: type[np.generic]) -> None:
    image = np.array([[0, 1]], dtype=dtype)

    result = IntensityTransformer().negative(image)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_float_outside_unit_range_requires_explicit_range() -> None:
    image = np.array([[-1.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="value_range explicitamente"):
        IntensityTransformer().negative(image)


@pytest.mark.parametrize(
    ("value_range", "exception"),
    [
        ([0, 1], TypeError),
        ((0,), TypeError),
        ((False, 1), TypeError),
        (("0", 1), TypeError),
        ((0, np.inf), ValueError),
        ((1, 1), ValueError),
        ((2, 1), ValueError),
        ((0, 0.5), ValueError),
    ],
)
def test_point_transform_rejects_invalid_value_range(
    value_range: object,
    exception: type[Exception],
) -> None:
    image = np.array([[0.0, 1.0]], dtype=np.float32)

    with pytest.raises(exception):
        IntensityTransformer().negative(
            image,
            value_range=value_range,  # type: ignore[arg-type]
        )
