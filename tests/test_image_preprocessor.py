import cv2 as cv
import numpy as np
import pytest

from dip_toolkit.modules.image_preprocessor import ImagePreprocessor


def test_correlate_identity_preserves_image_and_contract() -> None:
    image = np.arange(1, 10, dtype=np.uint8).reshape(3, 3)
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    result = ImagePreprocessor().correlate(image, kernel)

    assert np.array_equal(result, image)
    assert result.shape == image.shape
    assert result.dtype == np.float64


def test_correlate_asymmetric_kernel_is_not_convolution() -> None:
    image = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    kernel = np.array([[1, 2, 3], [0, 0, 0], [-1, -2, -3]], dtype=np.float64)
    expected = np.array(
        [[-23, -32, -17], [-30, -36, -18], [23, 32, 17]],
        dtype=np.float64,
    )

    correlation = ImagePreprocessor().correlate(image, kernel)
    convolution = ImagePreprocessor().correlate(image, np.flip(kernel))

    assert np.array_equal(correlation, expected)
    assert correlation[1, 1] == -36
    assert convolution[1, 1] == 36


def test_correlate_impulse_reveals_kernel_orientation() -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    image[2, 2] = 1
    kernel = np.arange(1, 10, dtype=np.float64).reshape(3, 3)

    result = ImagePreprocessor().correlate(image, kernel)

    assert np.array_equal(result[1:4, 1:4], np.flip(kernel))


def test_correlate_mean_kernel_has_known_border_and_center_values() -> None:
    image = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    kernel = np.ones((3, 3), dtype=np.float64) / 9

    result = ImagePreprocessor().correlate(image, kernel)

    assert result[1, 1] == pytest.approx(5.0)
    assert result[0, 0] == pytest.approx(12 / 9)


def test_correlate_does_not_modify_inputs() -> None:
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    original_image = image.copy()
    original_kernel = kernel.copy()

    result = ImagePreprocessor().correlate(image, kernel)

    assert np.array_equal(image, original_image)
    assert np.array_equal(kernel, original_kernel)
    assert not np.shares_memory(result, image)


@pytest.mark.parametrize(
    ("image", "exception"),
    [
        ([[1, 2]], TypeError),
        (np.array([], dtype=np.uint8), ValueError),
        (np.ones(3, dtype=np.uint8), ValueError),
        (np.ones((2, 2, 3), dtype=np.uint8), ValueError),
        (np.ones((2, 2), dtype=np.bool_), TypeError),
        (np.ones((2, 2), dtype=np.int64), TypeError),
        (np.array([[np.nan]], dtype=np.float32), ValueError),
        (np.array([[np.inf]], dtype=np.float64), ValueError),
    ],
)
def test_correlate_rejects_invalid_images(
    image: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ImagePreprocessor().correlate(  # type: ignore[arg-type]
            image,
            np.ones((3, 3)),
        )


@pytest.mark.parametrize(
    ("kernel", "exception"),
    [
        ([[1]], TypeError),
        (np.ones(3), ValueError),
        (np.ones((1, 1, 1)), ValueError),
        (np.empty((0, 3)), ValueError),
        (np.ones((2, 3)), ValueError),
        (np.ones((3, 2)), ValueError),
        (np.ones((3, 3), dtype=np.bool_), TypeError),
        (np.ones((3, 3), dtype=np.complex64), TypeError),
        (np.array([[np.nan]]), ValueError),
        (np.array([[np.inf]]), ValueError),
    ],
)
def test_correlate_rejects_invalid_kernels(
    kernel: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ImagePreprocessor().correlate(
            np.ones((3, 3), dtype=np.uint8),
            kernel,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("method_name", ["mean_filter", "gaussian_filter"])
def test_smoothing_preserves_constant_grayscale(method_name: str) -> None:
    image = np.full((5, 7), 42, dtype=np.uint8)
    method = getattr(ImagePreprocessor(), method_name)

    result = method(image, kernel_size=3)

    assert np.allclose(result, 42)
    assert result.dtype == np.float64


def test_smoothing_processes_color_channels_independently() -> None:
    image = np.empty((4, 5, 3), dtype=np.uint8)
    image[:] = (10, 40, 200)

    result = ImagePreprocessor().gaussian_filter(image, kernel_size=3, sigma=1.2)

    assert result.shape == image.shape
    assert np.allclose(result, image)
    assert result.dtype == np.float64


def test_median_removes_isolated_impulse_and_preserves_dtype() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255

    result = ImagePreprocessor().median_filter(image, kernel_size=3)

    assert np.count_nonzero(result) == 0
    assert result.dtype == image.dtype
    assert result.shape == image.shape


def test_sharpen_increases_local_response_without_uint8_overflow() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 200

    result = ImagePreprocessor().sharpen(image, amount=1.0)

    assert result[2, 2] == 1000
    assert result.dtype == np.float64
    assert result.shape == image.shape


def test_high_level_filters_do_not_modify_input() -> None:
    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    original = image.copy()
    preprocessor = ImagePreprocessor()

    preprocessor.mean_filter(image)
    preprocessor.gaussian_filter(image)
    preprocessor.median_filter(image)
    preprocessor.sharpen(image)

    assert np.array_equal(image, original)


@pytest.mark.parametrize(
    ("method_name", "arguments", "exception"),
    [
        ("mean_filter", {"kernel_size": 0}, ValueError),
        ("mean_filter", {"kernel_size": 2}, ValueError),
        ("mean_filter", {"kernel_size": True}, TypeError),
        ("gaussian_filter", {"sigma": 0}, ValueError),
        ("gaussian_filter", {"sigma": np.inf}, ValueError),
        ("gaussian_filter", {"sigma": "1"}, TypeError),
        ("median_filter", {"kernel_size": -1}, ValueError),
        ("median_filter", {"kernel_size": 4}, ValueError),
        ("sharpen", {"amount": 0}, ValueError),
        ("sharpen", {"amount": np.nan}, ValueError),
    ],
)
def test_filters_reject_invalid_parameters(
    method_name: str,
    arguments: dict[str, object],
    exception: type[Exception],
) -> None:
    method = getattr(ImagePreprocessor(), method_name)

    with pytest.raises(exception):
        method(np.ones((3, 3), dtype=np.uint8), **arguments)


def test_legacy_apply_filter_blur_matches_opencv() -> None:
    image = np.arange(49, dtype=np.uint8).reshape(7, 7)

    result = ImagePreprocessor().apply_filter(image, "blur", ksize=3)
    expected = cv.GaussianBlur(image, (3, 3), 0)

    assert np.array_equal(result, expected)
    assert result.dtype == image.dtype


def test_legacy_apply_filter_median_promotes_even_kernel_size() -> None:
    image = np.array(
        [
            [9, 1, 8, 2, 7, 3, 6],
            [4, 5, 0, 9, 1, 8, 2],
            [7, 3, 6, 4, 5, 0, 9],
            [1, 8, 2, 7, 3, 6, 4],
            [5, 0, 9, 1, 8, 2, 7],
            [3, 6, 4, 5, 0, 9, 1],
            [8, 2, 7, 3, 6, 4, 5],
        ],
        dtype=np.uint8,
    )

    result = ImagePreprocessor().apply_filter(image, "median", ksize=4)
    expected = cv.medianBlur(image, 5)

    assert np.array_equal(result, expected)
    assert result.dtype == image.dtype


def test_legacy_apply_filter_rejects_unknown_filter() -> None:
    with pytest.raises(ValueError, match="filter_type"):
        ImagePreprocessor().apply_filter(  # type: ignore[arg-type]
            np.ones((3, 3), dtype=np.uint8),
            "unknown",
        )
