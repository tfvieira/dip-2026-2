import numpy as np
import pytest

from dip_toolkit.modules.feature_extractor import FeatureExtractor


def test_sobel_x_responds_at_vertical_step() -> None:
    image = np.zeros((5, 7), dtype=np.uint8)
    image[:, 3:] = 10

    result = FeatureExtractor().sobel(image, direction="x")

    assert result.dtype == np.float64
    assert result.shape == image.shape
    assert result[2, 2] == 40
    assert result[2, 3] == 40
    assert result[2, 0] == 0


def test_sobel_y_and_magnitude_have_expected_direction() -> None:
    image = np.zeros((7, 5), dtype=np.float32)
    image[3:, :] = 2
    extractor = FeatureExtractor()

    gradient_y = extractor.sobel(image, direction="y")
    magnitude = extractor.sobel(image)

    assert gradient_y[2, 2] == 8
    assert gradient_y[3, 2] == 8
    assert magnitude[2, 2] == pytest.approx(8)


def test_laplacian_preserves_signed_step_response() -> None:
    image = np.zeros((5, 7), dtype=np.uint8)
    image[:, 3:] = 10

    result = FeatureExtractor().laplacian(image)

    assert result[2, 2] == 10
    assert result[2, 3] == -10
    assert result.dtype == np.float64
    assert result.shape == image.shape


def test_sobel_and_laplacian_do_not_modify_input() -> None:
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    original = image.copy()
    extractor = FeatureExtractor()

    extractor.sobel(image)
    extractor.laplacian(image)

    assert np.array_equal(image, original)


def test_canny_accepts_explicit_parameters_and_returns_binary_image() -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 255
    original = image.copy()

    result = FeatureExtractor().canny(
        image,
        low_threshold=50,
        high_threshold=150,
        aperture_size=3,
        l2_gradient=True,
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)) <= {0, 255}
    assert np.count_nonzero(result) > 0
    assert np.array_equal(image, original)


def test_extract_edges_preserves_legacy_defaults() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)
    image[:, 8:] = 255

    result = FeatureExtractor().extract_edges(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_canny_constant_image_has_no_artificial_border() -> None:
    image = np.full((8, 8), 100, dtype=np.uint8)

    result = FeatureExtractor().canny(image, 10, 20)

    assert np.count_nonzero(result) == 0


@pytest.mark.parametrize("direction", ["horizontal", "X", ""])
def test_sobel_rejects_invalid_direction(direction: str) -> None:
    with pytest.raises(ValueError, match="direction"):
        FeatureExtractor().sobel(
            np.ones((3, 3), dtype=np.uint8),
            direction=direction,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("image", "method_name", "arguments", "exception"),
    [
        (np.ones((3, 3, 3), dtype=np.uint8), "sobel", {}, ValueError),
        (np.ones((3, 3, 3), dtype=np.uint8), "laplacian", {}, ValueError),
        (
            np.ones((3, 3, 3), dtype=np.uint8),
            "canny",
            {"low_threshold": 1, "high_threshold": 2},
            ValueError,
        ),
        (
            np.ones((3, 3), dtype=np.float32),
            "canny",
            {"low_threshold": 1, "high_threshold": 2},
            TypeError,
        ),
    ],
)
def test_edge_operations_document_color_and_dtype_rejections(
    image: np.ndarray,
    method_name: str,
    arguments: dict[str, object],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        getattr(FeatureExtractor(), method_name)(image, **arguments)


@pytest.mark.parametrize(
    ("arguments", "exception"),
    [
        ({"low_threshold": 100, "high_threshold": 100}, ValueError),
        ({"low_threshold": 200, "high_threshold": 100}, ValueError),
        ({"low_threshold": -1, "high_threshold": 100}, ValueError),
        ({"low_threshold": np.nan, "high_threshold": 100}, ValueError),
        ({"low_threshold": 10, "high_threshold": np.inf}, ValueError),
        ({"low_threshold": "10", "high_threshold": 100}, TypeError),
        ({"low_threshold": 10, "high_threshold": 100, "aperture_size": 4}, ValueError),
        (
            {"low_threshold": 10, "high_threshold": 100, "aperture_size": 3.0},
            ValueError,
        ),
        ({"low_threshold": 10, "high_threshold": 100, "l2_gradient": 1}, TypeError),
    ],
)
def test_canny_rejects_invalid_parameters(
    arguments: dict[str, object],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        FeatureExtractor().canny(
            np.ones((8, 8), dtype=np.uint8),
            **arguments,  # type: ignore[arg-type]
        )
