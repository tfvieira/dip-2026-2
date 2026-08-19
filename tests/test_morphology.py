import numpy as np
import pytest

from dip_toolkit.modules.morphology import MorphologyProcessor


def test_create_structuring_elements_have_expected_shapes_and_values() -> None:
    processor = MorphologyProcessor()

    rectangle = processor.create_structuring_element("rectangle", (3, 3))
    ellipse = processor.create_structuring_element("ellipse", (3, 3))
    cross = processor.create_structuring_element("cross", (3, 3))

    assert np.array_equal(rectangle, np.ones((3, 3), dtype=np.uint8))
    expected_cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    assert np.array_equal(ellipse, expected_cross)
    assert np.array_equal(cross, expected_cross)


def test_erosion_and_dilation_match_small_manual_masks() -> None:
    processor = MorphologyProcessor()
    element = processor.create_structuring_element("rectangle", (3, 3))
    center_pixel = np.zeros((5, 5), dtype=np.uint8)
    center_pixel[2, 2] = 255
    filled_square = np.zeros((5, 5), dtype=np.uint8)
    filled_square[1:4, 1:4] = 255

    assert np.array_equal(
        processor.erode(center_pixel, element), np.zeros((5, 5), dtype=np.uint8)
    )
    assert np.array_equal(processor.dilate(center_pixel, element), filled_square)
    expected_eroded = np.zeros((5, 5), dtype=np.uint8)
    expected_eroded[2, 2] = 255
    assert np.array_equal(processor.erode(filled_square, element), expected_eroded)


def test_opening_removes_small_object_and_closing_fills_gap() -> None:
    processor = MorphologyProcessor()
    element = processor.create_structuring_element("rectangle", (3, 3))
    noise = np.zeros((7, 7), dtype=np.uint8)
    noise[3, 3] = 255
    with_gap = np.zeros((7, 7), dtype=np.uint8)
    with_gap[2:5, 2:5] = 255
    with_gap[3, 3] = 0

    assert np.array_equal(
        processor.opening(noise, element), np.zeros((7, 7), dtype=np.uint8)
    )
    expected_closed = np.zeros((7, 7), dtype=np.uint8)
    expected_closed[2:5, 2:5] = 255
    assert np.array_equal(processor.closing(with_gap, element), expected_closed)


def test_gradient_is_dilation_minus_erosion() -> None:
    processor = MorphologyProcessor()
    element = processor.create_structuring_element("rectangle", (3, 3))
    square = np.zeros((7, 7), dtype=np.uint8)
    square[2:5, 2:5] = 255

    result = processor.gradient(square, element)

    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[1:6, 1:6] = 255
    expected[3, 3] = 0
    assert np.array_equal(result, expected)


def test_operations_preserve_mask_contract_and_do_not_mutate_input() -> None:
    processor = MorphologyProcessor()
    element = processor.create_structuring_element("cross", (3, 3))
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 255
    original = mask.copy()

    for operation in (
        processor.erode,
        processor.dilate,
        processor.opening,
        processor.closing,
        processor.gradient,
    ):
        result = operation(mask, element, iterations=1)
        assert result.shape == mask.shape
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})
    assert np.array_equal(mask, original)


def test_constant_border_and_iterations_are_applied() -> None:
    processor = MorphologyProcessor()
    element = processor.create_structuring_element("rectangle", (3, 3))
    corner_pixel = np.zeros((5, 5), dtype=np.uint8)
    corner_pixel[0, 0] = 255

    result = processor.dilate(corner_pixel, element, iterations=2, border="constant")

    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[:3, :3] = 255
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    ("mask", "element", "kwargs", "exception"),
    [
        (
            np.empty((0, 2), dtype=np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            {},
            ValueError,
        ),
        (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            {},
            ValueError,
        ),
        (
            np.zeros((2, 2), dtype=np.float32),
            np.ones((3, 3), dtype=np.uint8),
            {},
            TypeError,
        ),
        (
            np.full((2, 2), 1, dtype=np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            {},
            ValueError,
        ),
        (
            np.zeros((2, 2), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint8),
            {},
            ValueError,
        ),
        (
            np.zeros((2, 2), dtype=np.uint8),
            np.ones((2, 3), dtype=np.uint8),
            {},
            ValueError,
        ),
        (
            np.zeros((2, 2), dtype=np.uint8),
            np.ones((3, 3), dtype=np.int16),
            {},
            TypeError,
        ),
        (
            np.zeros((2, 2), dtype=np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            {"iterations": 0},
            ValueError,
        ),
        (
            np.zeros((2, 2), dtype=np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            {"anchor": (3, 0)},
            ValueError,
        ),
        (
            np.zeros((2, 2), dtype=np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            {"border": "wrap"},
            ValueError,
        ),
    ],
)
def test_erode_rejects_invalid_inputs(
    mask: np.ndarray,
    element: np.ndarray,
    kwargs: dict[str, object],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        MorphologyProcessor().erode(mask, element, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("shape", ["square", "", "RECTANGLE"])
def test_create_structuring_element_rejects_unknown_shape(shape: str) -> None:
    with pytest.raises(ValueError):
        MorphologyProcessor().create_structuring_element(shape)  # type: ignore[arg-type]


@pytest.mark.parametrize("size", [(2, 3), (3, 0), (3,), [3, 3]])
def test_create_structuring_element_rejects_invalid_size(size: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MorphologyProcessor().create_structuring_element("cross", size)  # type: ignore[arg-type]
