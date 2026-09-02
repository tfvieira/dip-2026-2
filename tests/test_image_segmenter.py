import numpy as np
import pytest

from dip_toolkit.modules.image_segmenter import ImageSegmenter
from dip_toolkit.modules.morphology import MorphologyProcessor


def test_global_threshold_applies_both_polarities_to_a_ramp() -> None:
    image = np.array([[0, 50, 100, 150, 200]], dtype=np.uint8)
    segmenter = ImageSegmenter()
    assert np.array_equal(
        segmenter.global_threshold(image, 100),
        np.array([[0, 0, 255, 255, 255]], dtype=np.uint8),
    )
    assert np.array_equal(
        segmenter.global_threshold(image, 100, polarity="dark"),
        np.array([[255, 255, 255, 0, 0]], dtype=np.uint8),
    )


def test_otsu_separates_a_bimodal_image() -> None:
    result = ImageSegmenter().otsu(np.array([[10, 10, 200, 200]], dtype=np.uint8))
    assert 10 <= result.threshold < 200
    assert np.array_equal(result.mask, np.array([[0, 0, 255, 255]], dtype=np.uint8))


def test_adaptive_threshold_finds_bright_pixels_over_uneven_background() -> None:
    image = np.tile(np.linspace(20, 180, 9, dtype=np.uint8), (5, 1))
    image[2, [2, 5, 8]] = 220
    result = ImageSegmenter().adaptive_threshold(image, block_size=3, constant=10)
    assert np.array_equal(result[2, [2, 5, 8]], np.array([255, 255, 255]))
    assert set(np.unique(result)) <= {0, 255}


def test_edge_segmentation_reuses_canny() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)
    image[4:12, 4:12] = 255
    result = ImageSegmenter().edge_segmentation(image, 50, 150)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)) <= {0, 255}
    assert np.count_nonzero(result) > 0


def test_refine_mask_composes_dip_09_morphology() -> None:
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    mask[3, 3] = 0
    element = MorphologyProcessor().create_structuring_element("rectangle", (3, 3))
    result = ImageSegmenter().refine_mask(mask, element, ["closing"])
    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[2:5, 2:5] = 255
    assert np.array_equal(result, expected)


def test_connected_components_distinguishes_diagonal_connectivity() -> None:
    mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    segmenter = ImageSegmenter()
    four = segmenter.connected_components(mask, connectivity=4)
    eight = segmenter.connected_components(mask, connectivity=8)
    assert four.region_ids == (1, 2)
    assert eight.region_ids == (1,)
    assert four.labels[0, 0] != four.labels[1, 1]
    assert eight.labels[0, 0] == eight.labels[1, 1] == 1


def test_connected_components_are_deterministic_and_keep_background_zero() -> None:
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[0, 3] = 255
    mask[2, 1] = 255
    segmenter = ImageSegmenter()
    first = segmenter.connected_components(mask, connectivity=4)
    second = segmenter.connected_components(mask, connectivity=4)
    assert first.region_ids == (1, 2)
    assert np.array_equal(first.labels, second.labels)
    assert first.labels[0, 0] == 0
    assert first.labels[0, 3] == 1
    assert first.labels[2, 1] == 2


@pytest.mark.parametrize(
    ("call", "exception"),
    [
        (
            lambda: ImageSegmenter().global_threshold(
                np.ones((2, 2, 3), dtype=np.uint8), 1
            ),
            ValueError,
        ),
        (lambda: ImageSegmenter().otsu(np.ones((2, 2), dtype=np.float32)), TypeError),
        (
            lambda: ImageSegmenter().global_threshold(
                np.ones((2, 2), dtype=np.uint8), 256
            ),
            ValueError,
        ),
        (
            lambda: ImageSegmenter().adaptive_threshold(
                np.ones((3, 3), dtype=np.uint8), 2
            ),
            ValueError,
        ),
        (
            lambda: ImageSegmenter().adaptive_threshold(
                np.ones((3, 3), dtype=np.uint8), 3, np.nan
            ),
            ValueError,
        ),
        (
            lambda: ImageSegmenter().edge_segmentation(
                np.ones((3, 3), dtype=np.uint8), 50, 50
            ),
            ValueError,
        ),
        (
            lambda: ImageSegmenter().connected_components(
                np.ones((3, 3), dtype=np.uint8)
            ),
            ValueError,
        ),
    ],
)
def test_invalid_inputs_are_rejected(call: object, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        call()  # type: ignore[operator]
