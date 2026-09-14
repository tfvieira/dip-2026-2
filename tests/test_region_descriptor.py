import numpy as np
import pytest

from dip_toolkit.modules.image_segmenter import ImageSegmenter
from dip_toolkit.modules.region_descriptor import RegionDescriptorExtractor


def test_rectangle_has_expected_basic_descriptors_and_moments() -> None:
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:4, 2:6] = 255
    descriptor = RegionDescriptorExtractor().describe_region(mask, region_id=7)
    assert descriptor.region_id == 7
    assert descriptor.area == 12
    assert descriptor.centroid == (3.5, 2.0)
    assert descriptor.bounding_box == (2, 1, 4, 3)
    assert descriptor.aspect_ratio == pytest.approx(4 / 3)
    assert descriptor.perimeter == pytest.approx(10)
    assert descriptor.moments == pytest.approx((12, 42, 24, 15, 8, 0))


def test_approximate_disk_has_high_circularity() -> None:
    rows, columns = np.ogrid[:15, :15]
    mask = (((rows - 7) ** 2 + (columns - 7) ** 2) <= 25).astype(np.uint8) * 255
    descriptor = RegionDescriptorExtractor().describe_region(mask)
    assert descriptor.circularity > 0.8
    assert descriptor.centroid == (7.0, 7.0)


def test_shifted_object_updates_centroid_and_moments() -> None:
    first = np.zeros((7, 7), dtype=np.uint8)
    first[1:3, 1:3] = 255
    shifted = np.zeros((7, 7), dtype=np.uint8)
    shifted[3:5, 4:6] = 255
    extractor = RegionDescriptorExtractor()
    first_descriptor = extractor.describe_region(first)
    shifted_descriptor = extractor.describe_region(shifted)
    assert first_descriptor.centroid == (1.5, 1.5)
    assert shifted_descriptor.centroid == (4.5, 3.5)
    assert shifted_descriptor.moments.m10 > first_descriptor.moments.m10
    assert shifted_descriptor.moments.m01 > first_descriptor.moments.m01


def test_multiple_regions_follow_dip_10_ids_and_are_deterministic() -> None:
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    mask[3:5, 4:6] = 255
    regions = ImageSegmenter().connected_components(mask, connectivity=4)
    extractor = RegionDescriptorExtractor()
    first = extractor.describe_regions(regions)
    second = extractor.describe_regions(regions)
    assert tuple(item.region_id for item in first) == (1, 2)
    assert tuple(item.region_id for item in second) == (1, 2)
    assert first[0].bounding_box == (1, 1, 2, 2)
    assert first[1].bounding_box == (4, 3, 2, 2)


def test_describe_mask_and_contour_do_not_modify_input() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 255
    original = mask.copy()
    extractor = RegionDescriptorExtractor()
    assert len(extractor.describe_mask(mask)) == 1
    contour = extractor.extract_contour(mask)
    assert contour.dtype == np.int32
    assert contour.shape[1:] == (1, 2)
    assert np.array_equal(mask, original)


def test_single_pixel_has_defined_degenerate_perimeter_and_circularity() -> None:
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 255
    descriptor = RegionDescriptorExtractor().describe_region(mask)
    assert descriptor.perimeter == 0
    assert descriptor.circularity == 0
    assert descriptor.centroid == (1.0, 1.0)


@pytest.mark.parametrize(
    ("call", "exception"),
    [
        (
            lambda: RegionDescriptorExtractor().describe_region(
                np.empty((0, 2), dtype=np.uint8)
            ),
            ValueError,
        ),
        (
            lambda: RegionDescriptorExtractor().describe_region(
                np.ones((2, 2), dtype=np.float32)
            ),
            TypeError,
        ),
        (
            lambda: RegionDescriptorExtractor().describe_region(
                np.ones((2, 2), dtype=np.uint8)
            ),
            ValueError,
        ),
        (
            lambda: RegionDescriptorExtractor().describe_region(
                np.zeros((2, 2), dtype=np.uint8)
            ),
            ValueError,
        ),
        (
            lambda: RegionDescriptorExtractor().describe_labels(
                np.ones((2, 2), dtype=np.float32)
            ),
            TypeError,
        ),
        (
            lambda: RegionDescriptorExtractor().describe_labels(
                np.array([[0, -1]], dtype=np.int32)
            ),
            ValueError,
        ),
        (
            lambda: RegionDescriptorExtractor().describe_labels(
                np.array([[0, 1]], dtype=np.int32), [2]
            ),
            ValueError,
        ),
        (
            lambda: RegionDescriptorExtractor().describe_labels(
                np.array([[0, 1]], dtype=np.int32), [1, 1]
            ),
            ValueError,
        ),
    ],
)
def test_invalid_inputs_raise_clear_errors(
    call: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        call()  # type: ignore[operator]
