import numpy as np
import pytest

from dip_toolkit.modules.image_creator import ImageCreator


@pytest.mark.parametrize("shape", [(4, 6), [4, 6, 3]])
def test_create_filled_image_preserves_shape_and_value(
    shape: tuple[int, ...] | list[int],
) -> None:
    image = ImageCreator().create_filled_image(shape, value=42, dtype=np.uint8)

    assert image.shape == tuple(shape)
    assert image.dtype == np.uint8
    assert np.all(image == 42)


def test_create_filled_float_image_preserves_dtype() -> None:
    image = ImageCreator().create_filled_image(
        (3, 5),
        value=0.25,
        dtype=np.float32,
    )

    assert image.dtype == np.float32
    assert np.all(image == np.float32(0.25))


def test_create_zeros_and_ones_use_dtype_levels() -> None:
    creator = ImageCreator()

    zeros = creator.create_zeros_image([2, 3, 4], dtype=np.float64)
    integer_ones = creator.create_ones_image((2, 3), dtype=np.uint8)
    float_ones = creator.create_ones_image([2, 3], dtype=np.float32)

    assert np.array_equal(zeros, np.zeros((2, 3, 4), dtype=np.float64))
    assert np.all(integer_ones == 255)
    assert integer_ones.dtype == np.uint8
    assert np.all(float_ones == 1.0)
    assert float_ones.dtype == np.float32


@pytest.mark.parametrize(
    ("shape", "exception"),
    [
        ("3,4", TypeError),
        (3, TypeError),
        ((3,), ValueError),
        ((3, 4, 2, 1), ValueError),
        ((0, 4), ValueError),
        ((3, -1), ValueError),
        ((3, 2.5), TypeError),
        ((True, 3), TypeError),
    ],
)
def test_creation_rejects_invalid_shapes(
    shape: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ImageCreator().create_zeros_image(shape)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [np.bool_, np.complex64, np.str_, object, None])
def test_creation_rejects_non_real_dtypes(dtype: object) -> None:
    with pytest.raises(TypeError, match="dtype"):
        ImageCreator().create_zeros_image((3, 4), dtype=dtype)


@pytest.mark.parametrize(
    ("value", "dtype", "exception"),
    [
        (1.5, np.uint8, ValueError),
        (-1, np.uint8, ValueError),
        (256, np.uint8, ValueError),
        (np.inf, np.float32, ValueError),
        ("1", np.float32, TypeError),
    ],
)
def test_create_filled_image_rejects_invalid_values(
    value: object,
    dtype: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ImageCreator().create_filled_image(
            (2, 2),
            value=value,  # type: ignore[arg-type]
            dtype=dtype,
        )


@pytest.mark.parametrize("distribution", ["uniform", "normal", "rayleigh"])
def test_random_uint8_distributions_respect_dtype_and_interval(
    distribution: str,
) -> None:
    image = ImageCreator().create_random_image(
        (100, 80),
        distribution=distribution,  # type: ignore[arg-type]
        dtype=np.uint8,
        seed=7,
    )

    assert image.dtype == np.uint8
    assert image.min() >= 0
    assert image.max() <= 255


@pytest.mark.parametrize("distribution", ["uniform", "normal", "rayleigh"])
def test_random_float_distributions_respect_dtype_and_interval(
    distribution: str,
) -> None:
    image = ImageCreator().create_random_image(
        (100, 80),
        distribution=distribution,  # type: ignore[arg-type]
        dtype=np.float32,
        seed=7,
    )

    assert image.dtype == np.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_uniform_distribution_respects_explicit_interval() -> None:
    image = ImageCreator(seed=3).create_random_image(
        (50, 50),
        low=10,
        high=20,
    )

    assert image.min() >= 10
    assert image.max() < 20


def test_normal_and_rayleigh_accept_explicit_parameters() -> None:
    creator = ImageCreator(seed=10)

    normal = creator.create_random_image(
        (10, 10),
        "normal",
        np.float64,
        mean=0.4,
        std=0.05,
    )
    rayleigh = creator.create_random_image(
        (10, 10),
        "rayleigh",
        np.float64,
        scale=0.2,
    )

    assert normal.shape == (10, 10)
    assert rayleigh.shape == (10, 10)
    assert np.all((normal >= 0.0) & (normal <= 1.0))
    assert np.all((rayleigh >= 0.0) & (rayleigh <= 1.0))


def test_random_image_is_reproducible_with_per_call_seed() -> None:
    creator = ImageCreator()

    first = creator.create_random_image((8, 9, 3), seed=1234)
    second = creator.create_random_image((8, 9, 3), seed=1234)

    assert np.array_equal(first, second)


def test_random_image_is_reproducible_with_constructor_seed() -> None:
    first_creator = ImageCreator(seed=1234)
    second_creator = ImageCreator(seed=1234)

    first = first_creator.create_random_image((8, 9), "normal")
    second = second_creator.create_random_image((8, 9), "normal")

    assert np.array_equal(first, second)


def test_random_image_accepts_numpy_generator() -> None:
    first_rng = np.random.default_rng(99)
    second_rng = np.random.default_rng(99)

    first = ImageCreator(rng=first_rng).create_random_image((4, 5))
    second = ImageCreator(rng=second_rng).create_random_image((4, 5))

    assert np.array_equal(first, second)


def test_random_generation_does_not_change_numpy_global_state() -> None:
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)

    ImageCreator(seed=10).create_random_image((3, 4))
    observed = np.random.random()

    assert observed == expected


@pytest.mark.parametrize("distribution", ["gaussian", "", "Uniform"])
def test_random_image_rejects_invalid_distribution(distribution: str) -> None:
    with pytest.raises(ValueError, match="Distribuição inválida"):
        ImageCreator().create_random_image(
            (3, 4),
            distribution=distribution,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        ({"low": 20, "high": 10}, "low deve ser menor"),
        ({"low": -1, "high": 10}, "low deve ser"),
        ({"low": 0.5, "high": 10}, "devem ser inteiros"),
        ({"low": 0, "high": 257}, "high deve ser"),
        ({"mean": 0.5}, "não se aplica"),
    ],
)
def test_uniform_distribution_rejects_invalid_parameters(
    parameters: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ImageCreator().create_random_image((3, 4), **parameters)


@pytest.mark.parametrize(
    ("distribution", "parameters", "match"),
    [
        ("normal", {"std": 0}, "std deve ser"),
        ("normal", {"mean": -1}, "mean deve ser"),
        ("normal", {"scale": 1}, "não se aplica"),
        ("rayleigh", {"scale": 0}, "scale deve ser"),
        ("rayleigh", {"std": 1}, "não se aplica"),
    ],
)
def test_distributions_reject_invalid_parameters(
    distribution: str,
    parameters: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ImageCreator().create_random_image(
            (3, 4),
            distribution=distribution,  # type: ignore[arg-type]
            **parameters,
        )


def test_seed_and_rng_are_mutually_exclusive() -> None:
    rng = np.random.default_rng(1)

    with pytest.raises(ValueError, match="não ambos"):
        ImageCreator(seed=1, rng=rng)

    with pytest.raises(ValueError, match="não ambos"):
        ImageCreator().create_random_image((2, 2), seed=1, rng=rng)


def test_invalid_rng_is_rejected() -> None:
    with pytest.raises(TypeError, match="rng"):
        ImageCreator(rng=np.random.RandomState(1))  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_invalid_seed_is_rejected(seed: object) -> None:
    with pytest.raises((TypeError, ValueError), match="seed"):
        ImageCreator(seed=seed)  # type: ignore[arg-type]


def test_change_image_dtype_returns_copy_without_mutating_input() -> None:
    image = np.array([[0.0, 1.9], [2.1, 3.0]], dtype=np.float64)
    original = image.copy()

    converted = ImageCreator().change_image_dtype(image, np.uint8)

    assert converted.dtype == np.uint8
    assert np.array_equal(converted, np.array([[0, 1], [2, 3]], dtype=np.uint8))
    assert np.array_equal(image, original)
    assert not np.shares_memory(converted, image)


@pytest.mark.parametrize(
    ("image", "dtype", "exception"),
    [
        ([[1, 2]], np.uint8, TypeError),
        (np.array([1, 2]), np.uint8, ValueError),
        (np.empty((0, 2)), np.uint8, ValueError),
        (np.ones((2, 2), dtype=np.complex64), np.float32, TypeError),
        (np.array([[np.nan]]), np.float32, ValueError),
        (np.array([[256]]), np.uint8, ValueError),
    ],
)
def test_change_image_dtype_rejects_invalid_inputs(
    image: object,
    dtype: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ImageCreator().change_image_dtype(
            image,  # type: ignore[arg-type]
            dtype,
        )


def test_salt_and_pepper_float_values_are_documented_levels() -> None:
    image = ImageCreator().create_salt_and_pepper_noise(
        20,
        30,
        salt_prob=0.2,
        pepper_prob=0.3,
        seed=12,
    )

    assert image.dtype == np.float64
    assert set(np.unique(image)) == {-1.0, 0.5, 1.0}


def test_salt_and_pepper_integer_values_are_documented_levels() -> None:
    image = ImageCreator(seed=4).create_salt_and_pepper_noise(
        30,
        20,
        salt_prob=0.2,
        pepper_prob=0.3,
        dtype=np.uint8,
    )

    assert image.dtype == np.uint8
    assert set(np.unique(image)) == {0, 127, 255}


def test_salt_and_pepper_is_reproducible_and_does_not_overlap_classes() -> None:
    creator = ImageCreator()

    first = creator.create_salt_and_pepper_noise(
        8,
        9,
        salt_prob=1.0,
        pepper_prob=0.0,
        seed=5,
    )
    second = creator.create_salt_and_pepper_noise(
        8,
        9,
        salt_prob=1.0,
        pepper_prob=0.0,
        seed=5,
    )

    assert np.array_equal(first, second)
    assert np.all(first == 1.0)


def test_salt_and_pepper_accepts_float32_probabilities_summing_to_one() -> None:
    image = ImageCreator(seed=2).create_salt_and_pepper_noise(
        5,
        7,
        salt_prob=np.float32(0.6),
        pepper_prob=np.float32(0.4),
    )

    assert 0.5 not in image


@pytest.mark.parametrize(
    ("salt_prob", "pepper_prob"),
    [
        (-0.1, 0.1),
        (0.1, -0.1),
        (1.1, 0.0),
        (0.0, 1.1),
        (0.6, 0.5),
        (np.nan, 0.0),
    ],
)
def test_salt_and_pepper_rejects_invalid_probabilities(
    salt_prob: float,
    pepper_prob: float,
) -> None:
    with pytest.raises(ValueError):
        ImageCreator().create_salt_and_pepper_noise(
            salt_prob=salt_prob,
            pepper_prob=pepper_prob,
        )
