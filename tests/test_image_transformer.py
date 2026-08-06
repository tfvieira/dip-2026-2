import numpy as np
import pytest

from dip_toolkit.modules.image_transformer import ImageTransformer


@pytest.mark.parametrize("interpolation", ["nearest", "bilinear", "bicubic"])
def test_resample_preserves_dtype_and_channels(interpolation: str) -> None:
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    resized = ImageTransformer().resample(
        image,
        (9, 11),
        interpolation=interpolation,  # type: ignore[arg-type]
    )

    assert resized.shape == (9, 11, 3)
    assert resized.dtype == np.uint8
    assert not np.shares_memory(resized, image)


def test_resample_preserves_grayscale_dtype() -> None:
    image = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)

    resized = ImageTransformer().resample(image, (6, 8), "nearest")

    assert resized.shape == (6, 8)
    assert resized.dtype == np.float32


def test_quantize_uniform_produces_expected_uint8_levels() -> None:
    image = np.array([[0, 64, 128, 255]], dtype=np.uint8)
    original = image.copy()

    quantized = ImageTransformer().quantize_uniform(image, levels=4)

    assert np.array_equal(
        quantized,
        np.array([[0, 85, 170, 255]], dtype=np.uint8),
    )
    assert np.array_equal(image, original)
    assert quantized.dtype == np.uint8
    assert set(np.unique(quantized)) == {0, 85, 170, 255}


def test_quantize_uniform_uses_explicit_float_range() -> None:
    image = np.array([[0.0, 0.2, 0.6, 1.0]], dtype=np.float32)

    quantized = ImageTransformer().quantize_uniform(
        image,
        levels=3,
        value_range=(0.0, 1.0),
    )

    assert quantized.dtype == np.float32
    assert np.array_equal(quantized, np.array([[0.0, 0.0, 0.5, 1.0]]))


@pytest.mark.parametrize(
    ("image", "exception"),
    [
        (np.array([]), ValueError),
        (np.ones((3,)), ValueError),
        (np.ones((3, 4, 2)), ValueError),
        (np.ones((3, 4), dtype=np.bool_), TypeError),
        (np.ones((3, 4), dtype=np.int64), TypeError),
    ],
)
def test_transformer_rejects_invalid_images(
    image: np.ndarray,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ImageTransformer().resample(image, (4, 4))


@pytest.mark.parametrize(
    ("output_shape", "interpolation", "exception"),
    [
        ((0, 4), "nearest", ValueError),
        ((4, -1), "nearest", ValueError),
        ([4, 4], "nearest", TypeError),
        ((4, 4), "lanczos", ValueError),
    ],
)
def test_resample_rejects_invalid_arguments(
    output_shape: object,
    interpolation: object,
    exception: type[Exception],
) -> None:
    image = np.ones((3, 4), dtype=np.uint8)

    with pytest.raises(exception):
        ImageTransformer().resample(
            image,
            output_shape,  # type: ignore[arg-type]
            interpolation,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("levels", "value_range", "exception"),
    [
        (1, None, ValueError),
        (True, None, TypeError),
        (4, (1.0, 0.0), ValueError),
        (4, [0.0, 1.0], TypeError),
    ],
)
def test_quantize_uniform_rejects_invalid_arguments(
    levels: object,
    value_range: object,
    exception: type[Exception],
) -> None:
    image = np.ones((3, 4), dtype=np.uint8)

    with pytest.raises(exception):
        ImageTransformer().quantize_uniform(
            image,
            levels,  # type: ignore[arg-type]
            value_range=value_range,  # type: ignore[arg-type]
        )


def test_quantize_uniform_rejects_values_outside_range() -> None:
    image = np.array([[0, 255]], dtype=np.uint8)

    with pytest.raises(ValueError, match="fora de value_range"):
        ImageTransformer().quantize_uniform(
            image,
            levels=4,
            value_range=(0, 100),
        )


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_quantize_uniform_rejects_non_finite_values(invalid_value: float) -> None:
    image = np.array([[0.0, invalid_value]], dtype=np.float32)

    with pytest.raises(ValueError, match="valores finitos"):
        ImageTransformer().quantize_uniform(
            image,
            levels=4,
            value_range=(-1.0, 1.0),
        )
