import numpy as np
import pytest

from dip_toolkit.modules.image_analysis import ImageAnalysis


def test_grayscale_histogram_has_exact_counts_edges_and_pixel_sum() -> None:
    image = np.array([[0, 0], [1, 3]], dtype=np.uint8)
    original = image.copy()

    result = ImageAnalysis().compute_grayscale_histogram(
        image,
        bins=4,
        value_range=(0, 4),
    )

    assert np.array_equal(result.counts, np.array([2, 1, 0, 1]))
    assert np.array_equal(result.bin_edges, np.arange(5, dtype=np.float64))
    assert result.counts.shape == (4,)
    assert result.bin_edges.shape == (5,)
    assert result.counts.sum() == image.size
    assert np.array_equal(image, original)


def test_constant_grayscale_histogram_keeps_all_pixels() -> None:
    image = np.full((2, 3), 7, dtype=np.uint8)

    result = ImageAnalysis().compute_grayscale_histogram(image, bins=3)

    assert result.counts.sum() == image.size
    assert np.count_nonzero(result.counts) == 1


def test_rgb_histograms_are_labeled_and_distinguishable() -> None:
    image = np.tile(np.array([[[0, 1, 2]]], dtype=np.uint8), (2, 2, 1))

    results = ImageAnalysis().compute_color_histograms(
        image,
        channel_order="rgb",
        bins=4,
        value_range=(0, 4),
    )

    assert list(results) == ["r", "g", "b"]
    assert np.array_equal(results["r"].counts, [4, 0, 0, 0])
    assert np.array_equal(results["g"].counts, [0, 4, 0, 0])
    assert np.array_equal(results["b"].counts, [0, 0, 4, 0])
    assert all(result.counts.sum() == 4 for result in results.values())


def test_bgr_histograms_use_bgr_labels() -> None:
    image = np.tile(np.array([[[2, 1, 0]]], dtype=np.uint8), (2, 3, 1))

    results = ImageAnalysis().compute_color_histograms(
        image,
        channel_order="bgr",
        bins=4,
        value_range=(0, 4),
    )

    assert list(results) == ["b", "g", "r"]
    assert np.argmax(results["b"].counts) == 2
    assert np.argmax(results["g"].counts) == 1
    assert np.argmax(results["r"].counts) == 0
    assert all(result.counts.sum() == 6 for result in results.values())


@pytest.mark.parametrize("channel_order", ["gray", "RGB", "xyz", ""])
def test_color_histogram_rejects_unknown_channel_order(
    channel_order: str,
) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel_order"):
        ImageAnalysis().compute_color_histograms(
            image,
            channel_order=channel_order,  # type: ignore[arg-type]
        )


def test_color_histogram_rejects_grayscale() -> None:
    with pytest.raises(ValueError, match="imagens 3D"):
        ImageAnalysis().compute_color_histograms(
            np.zeros((2, 2), dtype=np.uint8),
            channel_order="rgb",
        )


def test_grayscale_histogram_rejects_color() -> None:
    with pytest.raises(ValueError, match="grayscale 2D"):
        ImageAnalysis().compute_grayscale_histogram(np.zeros((2, 2, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("bins", "exception"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (2.5, TypeError),
        ("4", TypeError),
    ],
)
def test_histogram_rejects_invalid_bins(
    bins: object,
    exception: type[Exception],
) -> None:
    image = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(exception):
        ImageAnalysis().compute_grayscale_histogram(
            image,
            bins=bins,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("value_range", "exception"),
    [
        ([0, 1], TypeError),
        ((0,), TypeError),
        ((False, 1), TypeError),
        ((0, np.inf), ValueError),
        ((1, 1), ValueError),
        ((2, 1), ValueError),
        ((0, 1), ValueError),
    ],
)
def test_histogram_rejects_invalid_range(
    value_range: object,
    exception: type[Exception],
) -> None:
    image = np.array([[0, 2]], dtype=np.uint8)

    with pytest.raises(exception):
        ImageAnalysis().compute_grayscale_histogram(
            image,
            value_range=value_range,  # type: ignore[arg-type]
        )


def test_cdf_has_known_normalized_values_and_is_non_decreasing() -> None:
    histogram = np.array([1, 2, 0, 1], dtype=np.int64)
    original = histogram.copy()

    cdf = ImageAnalysis().compute_cdf(histogram)

    assert cdf.dtype == np.float64
    assert np.allclose(cdf, [0.25, 0.75, 0.75, 1.0])
    assert np.all(np.diff(cdf) >= 0)
    assert cdf[-1] == pytest.approx(1.0)
    assert np.array_equal(histogram, original)


def test_cdf_without_normalization_returns_accumulated_counts() -> None:
    histogram = np.array([1.5, 0.5, 2.0])

    cdf = ImageAnalysis().compute_cdf(histogram, normalize=False)

    assert np.array_equal(cdf, np.array([1.5, 2.0, 4.0]))
    assert cdf.dtype == np.float64


def test_zero_sum_histogram_returns_zeros() -> None:
    histogram = np.zeros(4, dtype=np.int64)

    cdf = ImageAnalysis().compute_cdf(histogram)

    assert np.array_equal(cdf, np.zeros(4))
    assert cdf.dtype == np.float64


def test_constant_histogram_cdf_has_single_jump() -> None:
    histogram = np.array([0, 0, 6, 0], dtype=np.int64)

    cdf = ImageAnalysis().compute_cdf(histogram)

    assert np.array_equal(cdf, np.array([0.0, 0.0, 1.0, 1.0]))


@pytest.mark.parametrize(
    ("histogram", "exception"),
    [
        (np.array([[1, 2]]), ValueError),
        (np.array([1, -1]), ValueError),
        (np.array([1.0, np.nan]), ValueError),
        (np.array([1.0, np.inf]), ValueError),
        (np.array([True, False]), TypeError),
    ],
)
def test_cdf_rejects_invalid_histograms(
    histogram: np.ndarray,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ImageAnalysis().compute_cdf(histogram)


def test_legacy_compute_histogram_keeps_grayscale_shape() -> None:
    image = np.array([[0, 255]], dtype=np.uint8)

    histogram = ImageAnalysis().compute_histogram(image)

    assert histogram.shape == (256, 1)
    assert histogram.sum() == image.size


def test_legacy_compute_histogram_rejects_color_explicitly() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="compute_color_histograms"):
        ImageAnalysis().compute_histogram(image)
