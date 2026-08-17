import matplotlib
import numpy as np
import pytest

from dip_toolkit.modules.image_color_processor import ChannelSet, ColorImageProcessor

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def close_figures() -> None:
    yield
    plt.close("all")


def test_split_and_combine_preserve_bgr_image_and_metadata() -> None:
    image = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    processor = ColorImageProcessor()

    channels = processor.split_channels(image, "bgr")
    result = processor.combine_channels(channels)

    assert channels.channel_order == "bgr"
    assert np.array_equal(channels.channels[0], image[..., 0])
    assert np.array_equal(result, image)
    assert result is not image


@pytest.mark.parametrize("order", ["bgr", "rgb"])
def test_split_and_combine_round_trip_for_both_orders(order: str) -> None:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    processor = ColorImageProcessor()

    assert np.array_equal(
        processor.combine_channels(processor.split_channels(image, order)), image
    )


def test_rgb_bgr_conversion_swaps_known_primary_color_exactly() -> None:
    rgb_red = np.array([[[255, 0, 0]]], dtype=np.uint8)

    bgr_red = ColorImageProcessor().convert(rgb_red, "rgb", "bgr")

    assert np.array_equal(bgr_red, np.array([[[0, 0, 255]]], dtype=np.uint8))
    assert np.array_equal(ColorImageProcessor().convert(bgr_red, "bgr", "rgb"), rgb_red)


def test_color_space_conversions_have_documented_shapes_and_dtypes() -> None:
    bgr = np.array([[[0, 0, 255], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
    processor = ColorImageProcessor()

    for destination in ("hsv", "ycrcb", "lab"):
        converted = processor.convert(bgr, "bgr", destination)
        assert converted.shape == bgr.shape
        assert converted.dtype == np.uint8
    gray = processor.convert(bgr, "bgr", "gray")
    assert gray.shape == bgr.shape[:2]
    assert gray.dtype == np.uint8


def test_color_space_conversions_return_expected_values_for_primary_colors() -> None:
    bgr = np.array(
        [[[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255], [0, 0, 0]]],
        dtype=np.uint8,
    )
    processor = ColorImageProcessor()

    assert np.array_equal(
        processor.convert(bgr, "bgr", "hsv"),
        np.array(
            [[[0, 255, 255], [60, 255, 255], [120, 255, 255], [0, 0, 255], [0, 0, 0]]],
            dtype=np.uint8,
        ),
    )
    assert np.array_equal(
        processor.convert(bgr, "bgr", "ycrcb"),
        np.array(
            [
                [
                    [76, 255, 85],
                    [150, 21, 43],
                    [29, 107, 255],
                    [255, 128, 128],
                    [0, 128, 128],
                ]
            ],
            dtype=np.uint8,
        ),
    )
    assert np.array_equal(
        processor.convert(bgr, "bgr", "lab"),
        np.array(
            [
                [
                    [136, 208, 195],
                    [224, 42, 211],
                    [82, 207, 20],
                    [255, 128, 128],
                    [0, 128, 128],
                ]
            ],
            dtype=np.uint8,
        ),
    )
    assert np.array_equal(
        processor.convert(bgr, "bgr", "gray"),
        np.array([[76, 150, 29, 255, 0]], dtype=np.uint8),
    )


def test_selected_hsv_round_trip_is_within_uint8_tolerance() -> None:
    bgr = np.array([[[12, 34, 200], [255, 10, 0]]], dtype=np.uint8)
    processor = ColorImageProcessor()

    restored = processor.convert(processor.convert(bgr, "bgr", "hsv"), "hsv", "bgr")

    # OpenCV representa H em 180 níveis no HSV uint8; a quantização pode
    # introduzir pequenas diferenças em uma conversão de ida e volta.
    assert np.max(np.abs(restored.astype(int) - bgr.astype(int))) <= 3


def test_channel_histograms_have_declared_order_and_counts() -> None:
    image = np.array([[[2, 1, 0], [2, 1, 0]]], dtype=np.uint8)

    histograms = ColorImageProcessor().channel_histograms(image, "bgr")

    assert list(histograms) == ["b", "g", "r"]
    assert histograms["b"][2] == 2
    assert histograms["g"][1] == 2
    assert histograms["r"][0] == 2
    assert all(histogram.sum() == 2 for histogram in histograms.values())


def test_negative_operation_is_applied_independently_per_channel() -> None:
    image = np.array([[[0, 10, 255]]], dtype=np.uint8)

    result = ColorImageProcessor().apply_channel_operation(image, "bgr", "negative")

    assert np.array_equal(result, np.array([[[255, 245, 0]]], dtype=np.uint8))
    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_histogram_plot_is_composable_and_uses_channel_labels() -> None:
    processor = ColorImageProcessor()
    histograms = processor.channel_histograms(
        np.zeros((2, 2, 3), dtype=np.uint8), "rgb"
    )

    figure, axis = processor.plot_channel_histograms(histograms, "rgb")

    assert figure is axis.figure
    assert [line.get_label() for line in axis.lines] == ["R", "G", "B"]


def test_legacy_public_apis_remain_available(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = ColorImageProcessor()
    bgr_red = np.array([[[0, 0, 255]]], dtype=np.uint8)
    show_calls: list[None] = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(None))

    assert np.array_equal(
        processor.convert_color_space(bgr_red, "rgb_to_hsv"),
        processor.convert(bgr_red, "bgr", "hsv"),
    )
    assert np.array_equal(
        processor.rgb_to_cmyk(bgr_red),
        np.array([[[0, 254, 254, 0]]], dtype=np.uint8),
    )
    assert processor.plot_rgb_histograms(bgr_red) is None
    assert processor.plot_rgb_3d_cube(bgr_red) is None
    assert len(show_calls) == 2


@pytest.mark.parametrize(
    ("image", "order", "exception"),
    [
        (np.empty((0, 2, 3), dtype=np.uint8), "bgr", ValueError),
        (np.zeros((2, 2), dtype=np.uint8), "bgr", ValueError),
        (np.zeros((2, 2, 4), dtype=np.uint8), "bgr", ValueError),
        (np.zeros((2, 2, 3), dtype=np.float32), "bgr", TypeError),
        (np.zeros((2, 2, 3), dtype=np.uint8), "hsv", ValueError),
    ],
)
def test_split_rejects_invalid_color_images(
    image: np.ndarray,
    order: str,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ColorImageProcessor().split_channels(image, order)  # type: ignore[arg-type]


def test_combine_rejects_mismatched_channels() -> None:
    channels = ChannelSet(
        (
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
        ),
        "rgb",
    )

    with pytest.raises(ValueError, match="mesmo shape"):
        ColorImageProcessor().combine_channels(channels)


def test_combine_rejects_invalid_channel_count_and_dtype() -> None:
    processor = ColorImageProcessor()
    two_channels = ChannelSet(
        (np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)),  # type: ignore[arg-type]
        "bgr",
    )
    mixed_dtypes = ChannelSet(
        (
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.int16),
        ),
        "bgr",
    )

    with pytest.raises(ValueError, match="exatamente três"):
        processor.combine_channels(two_channels)
    with pytest.raises(TypeError, match="mesmo dtype"):
        processor.combine_channels(mixed_dtypes)


@pytest.mark.parametrize("space", ["xyz", "", "RGB"])
def test_convert_rejects_unknown_color_space(space: str) -> None:
    with pytest.raises(ValueError):
        ColorImageProcessor().convert(
            np.zeros((2, 2, 3), dtype=np.uint8),
            "bgr",
            space,  # type: ignore[arg-type]
        )


def test_convert_rejects_color_input_for_gray_source() -> None:
    with pytest.raises(ValueError, match="grayscale"):
        ColorImageProcessor().convert(
            np.zeros((2, 2, 3), dtype=np.uint8), "gray", "bgr"
        )


def test_channel_operation_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError, match="operation"):
        ColorImageProcessor().apply_channel_operation(
            np.zeros((2, 2, 3), dtype=np.uint8),
            "rgb",
            "equalize",  # type: ignore[arg-type]
        )
