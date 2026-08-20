import numpy as np
import pytest

from dip_toolkit.modules.fourier_transformer import FourierTransformer


def test_dft_returns_centered_complex_spectrum_without_modifying_image() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    original = image.copy()

    spectrum = FourierTransformer().dft(image)

    expected = np.fft.fftshift(np.fft.fft2(image))
    assert spectrum.shape == image.shape
    assert spectrum.dtype == np.complex128
    assert np.array_equal(spectrum, expected)
    assert np.array_equal(image, original)


def test_idft_reconstructs_float64_image_with_numerical_tolerance() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    transformer = FourierTransformer()
    spectrum = transformer.dft(image)
    original_spectrum = spectrum.copy()

    reconstructed = transformer.idft(spectrum)

    assert reconstructed.shape == image.shape
    assert reconstructed.dtype == np.float64
    assert np.isrealobj(reconstructed)
    assert np.allclose(reconstructed, image, atol=1e-12)
    assert np.array_equal(spectrum, original_spectrum)


def test_constant_image_concentrates_dc_at_center() -> None:
    image = np.full((4, 4), 5.0)

    spectrum = FourierTransformer().dft(image)

    center = (image.shape[0] // 2, image.shape[1] // 2)
    without_dc = spectrum.copy()
    without_dc[center] = 0
    assert spectrum[center] == pytest.approx(image.sum())
    assert np.allclose(without_dc, 0.0, atol=1e-12)


def test_impulse_has_constant_magnitude_spectrum() -> None:
    image = np.zeros((5, 5), dtype=np.float64)
    image[1, 3] = 1.0
    transformer = FourierTransformer()

    magnitude = transformer.magnitude(transformer.dft(image))

    assert np.allclose(magnitude, 1.0, atol=1e-12)


def test_magnitude_matches_absolute_values_without_logarithm() -> None:
    spectrum = np.array([[3 + 4j, 0j], [-5j, -8 + 6j]])

    magnitude = FourierTransformer().magnitude(spectrum)

    assert magnitude.shape == spectrum.shape
    assert magnitude.dtype == np.float64
    assert np.isrealobj(magnitude)
    assert np.array_equal(magnitude, np.array([[5.0, 0.0], [5.0, 10.0]]))
    assert np.all(magnitude >= 0)


def test_phase_matches_numpy_angle_and_is_finite() -> None:
    spectrum = np.array([[1 + 0j, 1j], [-1 + 0j, -1j]])

    phase = FourierTransformer().phase(spectrum)

    assert phase.shape == spectrum.shape
    assert phase.dtype == np.float64
    assert np.isrealobj(phase)
    assert np.all(np.isfinite(phase))
    assert np.array_equal(phase, np.angle(spectrum))


def test_low_pass_mask_is_binary_centered_and_radially_symmetric() -> None:
    mask = FourierTransformer().ideal_low_pass_mask((5, 5), cutoff=1)
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    assert mask.shape == (5, 5)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 1}
    assert mask[2, 2] == 1
    assert np.array_equal(mask, expected)
    assert np.array_equal(mask, np.flip(mask, axis=0))
    assert np.array_equal(mask, np.flip(mask, axis=1))


def test_high_pass_mask_is_exact_low_pass_complement() -> None:
    transformer = FourierTransformer()
    low_pass = transformer.ideal_low_pass_mask((5, 5), cutoff=1.5)
    high_pass = transformer.ideal_high_pass_mask((5, 5), cutoff=1.5)

    assert high_pass.dtype == np.uint8
    assert np.array_equal(high_pass, 1 - low_pass)


def test_apply_mask_multiplies_without_modifying_inputs() -> None:
    spectrum = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    original_spectrum = spectrum.copy()
    original_mask = mask.copy()

    filtered = FourierTransformer().apply_mask(spectrum, mask)

    assert filtered.shape == spectrum.shape
    assert np.iscomplexobj(filtered)
    assert np.array_equal(filtered, spectrum * mask)
    assert np.array_equal(spectrum, original_spectrum)
    assert np.array_equal(mask, original_mask)


def test_low_and_high_pass_flows_reconstruct_original_shape() -> None:
    image = np.arange(25, dtype=np.float64).reshape(5, 5)
    transformer = FourierTransformer()
    spectrum = transformer.dft(image)

    low_mask = transformer.ideal_low_pass_mask(spectrum.shape, cutoff=1)
    high_mask = transformer.ideal_high_pass_mask(spectrum.shape, cutoff=1)
    low_image = transformer.idft(transformer.apply_mask(spectrum, low_mask))
    high_image = transformer.idft(transformer.apply_mask(spectrum, high_mask))

    assert low_image.shape == image.shape
    assert high_image.shape == image.shape
    assert np.allclose(low_image + high_image, image, atol=1e-12)


@pytest.mark.parametrize(
    ("image", "exception"),
    [
        ([1, 2], TypeError),
        (np.array([1, 2]), ValueError),
        (np.ones((2, 2, 3)), ValueError),
        (np.empty((0, 2)), ValueError),
        (np.array([[np.nan]]), ValueError),
        (np.array([[np.inf]]), ValueError),
        (np.array([[1 + 2j]]), TypeError),
        (np.array([[True]]), TypeError),
        (np.array([["pixel"]]), TypeError),
    ],
)
def test_dft_rejects_invalid_images(
    image: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        FourierTransformer().dft(image)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("spectrum", "exception"),
    [
        ([[1j]], TypeError),
        (np.array([1j]), ValueError),
        (np.empty((0, 2), dtype=np.complex128), ValueError),
        (np.array([[complex(np.nan, 0)]]), ValueError),
        (np.array([[complex(0, np.inf)]]), ValueError),
        (np.ones((2, 2), dtype=np.float64), TypeError),
    ],
)
def test_spectrum_operations_reject_invalid_spectra(
    spectrum: object,
    exception: type[Exception],
) -> None:
    transformer = FourierTransformer()
    for operation in (transformer.idft, transformer.magnitude, transformer.phase):
        with pytest.raises(exception):
            operation(spectrum)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("shape", "cutoff", "exception"),
    [
        ([5, 5], 1, TypeError),
        ((5,), 1, TypeError),
        ((5, 5, 1), 1, TypeError),
        ((0, 5), 1, ValueError),
        ((5, -1), 1, ValueError),
        ((5.0, 5), 1, TypeError),
        ((True, 5), 1, TypeError),
        ((5, 5), "1", TypeError),
        ((5, 5), True, TypeError),
        ((5, 5), np.nan, ValueError),
        ((5, 5), np.inf, ValueError),
        ((5, 5), -1, ValueError),
        ((5, 5), 3, ValueError),
    ],
)
def test_low_pass_mask_rejects_invalid_shape_or_cutoff(
    shape: object,
    cutoff: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        FourierTransformer().ideal_low_pass_mask(  # type: ignore[arg-type]
            shape,
            cutoff,
        )


@pytest.mark.parametrize(
    ("mask", "exception"),
    [
        ([[1, 0], [0, 1]], TypeError),
        (np.array([1, 0]), ValueError),
        (np.empty((0, 2)), ValueError),
        (np.array([[1, 0], [0, 0.5]]), ValueError),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), ValueError),
        (np.array([[1 + 0j]]), TypeError),
        (np.array([[True, False]]), TypeError),
    ],
)
def test_apply_mask_rejects_invalid_masks(
    mask: object,
    exception: type[Exception],
) -> None:
    spectrum = np.ones((2, 2), dtype=np.complex128)

    with pytest.raises(exception):
        FourierTransformer().apply_mask(  # type: ignore[arg-type]
            spectrum,
            mask,
        )


def test_apply_mask_rejects_different_shape() -> None:
    spectrum = np.ones((2, 2), dtype=np.complex128)
    mask = np.ones((3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="mesmo shape"):
        FourierTransformer().apply_mask(spectrum, mask)
