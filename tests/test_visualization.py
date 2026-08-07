import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dip_toolkit.modules.visualization import Visualization


@pytest.fixture(autouse=True)
def close_figures() -> None:
    yield
    plt.close("all")


def test_plot_histogram_returns_figure_axis_title_and_label() -> None:
    counts = np.array([1, 3, 2])
    edges = np.array([0.0, 1.0, 2.0, 3.0])

    figure, axis = Visualization().plot_histogram(
        counts,
        edges,
        title="Histograma",
        label="Original",
    )

    assert figure is axis.figure
    assert axis.get_title() == "Histograma"
    assert len(axis.lines) == 1
    assert axis.lines[0].get_label() == "Original"
    assert np.array_equal(axis.lines[0].get_xdata(), [0.5, 1.5, 2.5])
    assert axis.get_legend() is not None


def test_plot_histogram_does_not_call_show(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_show() -> None:
        raise AssertionError("plt.show não deve ser chamado")

    monkeypatch.setattr(plt, "show", fail_show)

    Visualization().plot_histogram(np.array([1, 2, 1]))


def test_plot_histogram_composes_on_existing_axis() -> None:
    figure, axis = plt.subplots()
    visualization = Visualization()

    first_figure, first_axis = visualization.plot_histogram(
        np.array([1, 2]),
        ax=axis,
        label="Antes",
    )
    second_figure, second_axis = visualization.plot_histogram(
        np.array([2, 1]),
        ax=axis,
        label="Depois",
    )

    assert first_figure is figure
    assert second_figure is figure
    assert first_axis is axis
    assert second_axis is axis
    assert len(axis.lines) == 2
    assert [line.get_label() for line in axis.lines] == ["Antes", "Depois"]


def test_plot_histogram_accepts_legacy_column_shape() -> None:
    counts = np.array([[1], [2], [3]], dtype=np.float32)

    _, axis = Visualization().plot_histogram(counts)

    assert len(axis.lines) == 1
    assert np.array_equal(axis.lines[0].get_ydata(), [1, 2, 3])


@pytest.mark.parametrize(
    ("counts", "exception"),
    [
        ([1, 2], TypeError),
        (np.array([]), ValueError),
        (np.ones((2, 2)), ValueError),
        (np.array([1, -1]), ValueError),
        (np.array([1.0, np.nan]), ValueError),
        (np.array([True, False]), TypeError),
    ],
)
def test_plot_histogram_rejects_invalid_counts(
    counts: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        Visualization().plot_histogram(counts)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("edges", "exception"),
    [
        (np.array([[0, 1, 2]]), ValueError),
        (np.array([0, 1]), ValueError),
        (np.array([0, 2, 1, 3]), ValueError),
        (np.array([0.0, 1.0, np.nan, 3.0]), ValueError),
        (["0", "1", "2", "3"], TypeError),
    ],
)
def test_plot_histogram_rejects_invalid_edges(
    edges: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        Visualization().plot_histogram(
            np.array([1, 2, 3]),
            edges,  # type: ignore[arg-type]
        )


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


def test_show_image_uses_explicit_float_range_without_clipping() -> None:
    image = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)

    _, axis = Visualization().show_image(
        image,
        channel_order="gray",
        value_range=(-1.0, 1.0),
    )

    assert axis.images[0].get_clim() == (-1.0, 1.0)
    assert np.array_equal(axis.images[0].get_array(), image)


def test_show_image_normalizes_float_color_with_explicit_range() -> None:
    image = np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32)

    _, axis = Visualization().show_image(
        image,
        channel_order="bgr",
        value_range=(-1.0, 1.0),
    )

    assert np.allclose(axis.images[0].get_array(), np.array([[[1.0, 0.5, 0.0]]]))


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


def test_show_image_rejects_values_outside_explicit_range() -> None:
    image = np.array([[-1.0, 1.5]], dtype=np.float32)

    with pytest.raises(ValueError, match="fora de value_range"):
        Visualization().show_image(
            image,
            channel_order="gray",
            value_range=(-1.0, 1.0),
        )


def test_show_image_preserves_constant_float_color_in_unit_range() -> None:
    image = np.ones((2, 3, 3), dtype=np.float32)

    _, axis = Visualization().show_image(
        image,
        channel_order="rgb",
    )

    assert np.array_equal(axis.images[0].get_array(), image)


def test_show_image_requires_range_for_constant_float_color_outside_unit_range() -> (
    None
):
    image = np.full((2, 3, 3), -1.0, dtype=np.float32)

    with pytest.raises(ValueError, match="informar value_range"):
        Visualization().show_image(
            image,
            channel_order="rgb",
        )
