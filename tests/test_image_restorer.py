import numpy as np
import pytest

from dip_toolkit.modules.image_preprocessor import ImagePreprocessor
from dip_toolkit.modules.image_restorer import ImageRestorer


def test_gaussian_noise_is_reproducible_and_has_expected_statistics() -> None:
    image = np.full((400, 300), 100, dtype=np.uint8)
    restorer = ImageRestorer()
    first = restorer.add_gaussian_noise(image, mean=2, std=4, seed=17)
    second = restorer.add_gaussian_noise(image, mean=2, std=4, seed=17)

    assert np.array_equal(first, second)
    assert first.dtype == np.float64
    assert (first - image).mean() == pytest.approx(2, abs=0.05)
    assert (first - image).std() == pytest.approx(4, abs=0.05)
    assert np.array_equal(image, np.full((400, 300), 100, dtype=np.uint8))


def test_gaussian_noise_does_not_clip() -> None:
    image = np.full((2, 2), 250, dtype=np.uint8)
    assert np.all(
        ImageRestorer().add_gaussian_noise(image, mean=100, std=0.1, seed=1) > 255
    )


def test_salt_and_pepper_is_reproducible_and_exclusive() -> None:
    image = np.full((100, 100), 120, dtype=np.uint8)
    restorer = ImageRestorer()
    first = restorer.add_salt_and_pepper_noise(image, 0.2, 0.3, seed=7)
    second = restorer.add_salt_and_pepper_noise(image, 0.2, 0.3, seed=7)

    assert np.array_equal(first, second)
    assert set(np.unique(first)) <= {0.0, 120.0, 255.0}
    assert np.count_nonzero(first == 0) == pytest.approx(3000, abs=180)
    assert np.count_nonzero(first == 255) == pytest.approx(2000, abs=160)


def test_salt_and_pepper_uses_one_class_for_a_color_pixel() -> None:
    image = np.full((20, 20, 3), 80, dtype=np.uint8)
    result = ImageRestorer().add_salt_and_pepper_noise(image, 0.2, 0.2, seed=10)
    assert np.all(result[..., 0] == result[..., 1])
    assert np.all(result[..., 1] == result[..., 2])


def test_motion_blur_kernel_and_impulse_response() -> None:
    image = np.zeros((7, 7), dtype=np.uint8)
    image[3, 3] = 1
    restorer = ImageRestorer()
    kernel = restorer.motion_blur_kernel(3, "horizontal")
    result = restorer.apply_motion_blur(image, 3, "horizontal")
    expected = np.zeros((7, 7))
    expected[3, 2:5] = 1 / 3

    assert kernel.sum() == pytest.approx(1)
    assert np.allclose(result, expected)


def test_motion_blur_handles_color_channels_independently() -> None:
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[2, 2] = (3, 6, 9)
    result = ImageRestorer().apply_motion_blur(image, 3, "vertical")
    assert result.shape == image.shape
    assert np.allclose(result[1:4, 2], np.array([[1, 2, 3]] * 3))


def test_mse_has_known_value_and_accepts_filter_dtype() -> None:
    restorer = ImageRestorer()
    reference = np.zeros((2, 2), dtype=np.uint8)
    assert restorer.mse(reference, reference) == 0
    assert restorer.mse(reference, np.full((2, 2), 3.0)) == 9


def test_spatial_filters_improve_controlled_noise() -> None:
    image = np.full((31, 31), 100, dtype=np.uint8)
    restorer = ImageRestorer()
    filters = ImagePreprocessor()
    gaussian = restorer.add_gaussian_noise(image, std=20, seed=4)
    impulse = restorer.add_salt_and_pepper_noise(image, 0.05, 0.05, seed=4)
    assert restorer.mse(image, filters.gaussian_filter(gaussian, 3, 1)) < restorer.mse(
        image, gaussian
    )
    assert restorer.mse(image, filters.median_filter(impulse, 3)) < restorer.mse(
        image, impulse
    )


@pytest.mark.parametrize(
    ("method_name", "arguments", "exception"),
    [
        ("add_gaussian_noise", {"std": 0}, ValueError),
        ("add_gaussian_noise", {"mean": np.nan}, ValueError),
        (
            "add_salt_and_pepper_noise",
            {"salt_prob": 0.8, "pepper_prob": 0.3},
            ValueError,
        ),
        ("add_salt_and_pepper_noise", {"salt_prob": -0.1}, ValueError),
        ("add_gaussian_noise", {"seed": -1}, ValueError),
    ],
)
def test_noise_parameters_are_validated(
    method_name: str,
    arguments: dict[str, object],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        getattr(ImageRestorer(), method_name)(
            np.ones((3, 3), dtype=np.uint8), **arguments
        )


@pytest.mark.parametrize(
    "image",
    [
        np.array([], dtype=np.uint8),
        np.ones((3,), dtype=np.uint8),
        np.ones((3, 3, 2), dtype=np.uint8),
        np.ones((3, 3), dtype=np.int64),
        np.array([[np.nan]], dtype=np.float32),
    ],
)
def test_invalid_images_are_rejected(image: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        ImageRestorer().add_gaussian_noise(image)


def test_mse_rejects_different_shape() -> None:
    with pytest.raises(ValueError, match="mesmo shape"):
        ImageRestorer().mse(
            np.ones((2, 2), dtype=np.uint8), np.ones((3, 3), dtype=np.uint8)
        )
