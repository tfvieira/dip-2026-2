import numpy as np
import pytest

from dip_toolkit.modules.statistical_tools import StatisticalTools


def test_get_image_info_describes_grayscale_image() -> None:
    image = np.array([[0, 10], [20, 255]], dtype=np.uint8)

    info = StatisticalTools().get_image_info(image)

    assert info["height"] == 2
    assert info["width"] == 2
    assert info["ndim"] == 2
    assert info["channels"] == 1
    assert info["shape"] == (2, 2)
    assert info["dtype"] == np.dtype(np.uint8)
    assert info["nbytes"] == 4
    assert info["minimum"] == 0
    assert info["maximum"] == 255
    assert info["expected_range"] == (0, 255)


def test_get_image_info_describes_color_image() -> None:
    image = np.zeros((3, 5, 3), dtype=np.float32)
    image[0, 0] = (0.2, 0.5, 1.0)

    info = StatisticalTools().get_image_info(image)

    assert info["ndim"] == 3
    assert info["channels"] == 3
    assert info["shape"] == (3, 5, 3)
    assert info["dtype"] == np.dtype(np.float32)
    assert info["minimum"] == 0.0
    assert info["maximum"] == 1.0
    limits = np.finfo(np.float32)
    assert info["expected_range"] == (float(limits.min), float(limits.max))


def test_get_image_info_accepts_explicit_float_range() -> None:
    image = np.array([[-1.0, 0.5, 1.0]], dtype=np.float64)

    info = StatisticalTools().get_image_info(image, expected_range=(-1.0, 1.0))

    assert info["expected_range"] == (-1.0, 1.0)


@pytest.mark.parametrize(
    ("image", "exception"),
    [
        ("not an image", TypeError),
        (np.empty((0, 3), dtype=np.uint8), ValueError),
        (np.ones((4,), dtype=np.uint8), ValueError),
        (np.ones((2, 2, 4), dtype=np.uint8), ValueError),
        (np.ones((2, 2), dtype=np.bool_), TypeError),
    ],
)
def test_get_image_info_rejects_invalid_images(
    image: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        StatisticalTools().get_image_info(image)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("expected_range", "exception"),
    [
        ([0.0, 1.0], TypeError),
        ((1.0, 0.0), ValueError),
        ((float("nan"), 1.0), ValueError),
    ],
)
def test_get_image_info_rejects_invalid_expected_range(
    expected_range: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        StatisticalTools().get_image_info(
            np.ones((2, 2), dtype=np.float32),
            expected_range=expected_range,  # type: ignore[arg-type]
        )
