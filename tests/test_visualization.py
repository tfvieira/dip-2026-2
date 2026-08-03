import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from dip_toolkit.modules.visualization import Visualization


def test_show_image_returns_composable_figure_for_grayscale() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)

    figure, axis = Visualization().show_image(
        image,
        channel_order="gray",
        title="Grayscale",
    )

    assert figure is axis.figure
    assert axis.get_title() == "Grayscale"
    assert len(axis.images) == 1


def test_show_image_converts_bgr_for_matplotlib() -> None:
    image = np.array([[[0, 0, 255]]], dtype=np.uint8)

    _, axis = Visualization().show_image(image, channel_order="bgr")

    assert np.array_equal(axis.images[0].get_array(), np.array([[[255, 0, 0]]]))


def test_compare_images_returns_one_axis_per_image() -> None:
    first = np.zeros((2, 3), dtype=np.uint8)
    second = np.ones((2, 3), dtype=np.uint8)

    figure, axes = Visualization().compare_images(
        [first, second],
        ["Original", "Comparação"],
    )

    assert axes.shape == (2,)
    assert all(axis.figure is figure for axis in axes)
    assert [axis.get_title() for axis in axes] == ["Original", "Comparação"]


@pytest.mark.parametrize(
    ("image", "channel_order", "exception"),
    [
        (np.empty((0, 2), dtype=np.uint8), "gray", ValueError),
        (np.ones((2, 2, 2), dtype=np.uint8), "rgb", ValueError),
        (np.ones((2, 2), dtype=np.uint8), "bgr", ValueError),
        (np.ones((2, 2, 3), dtype=np.uint8), "gray", ValueError),
        (np.ones((2, 2), dtype=np.bool_), "gray", TypeError),
    ],
)
def test_show_image_rejects_invalid_images_or_channel_order(
    image: np.ndarray,
    channel_order: str,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        Visualization().show_image(
            image,
            channel_order=channel_order,  # type: ignore[arg-type]
        )


def test_compare_images_rejects_mismatched_titles() -> None:
    image = np.ones((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="mesmo tamanho"):
        Visualization().compare_images([image], ["Um", "Dois"])
