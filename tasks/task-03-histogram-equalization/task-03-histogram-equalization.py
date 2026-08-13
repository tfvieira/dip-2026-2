import numpy as np


def equalize_histogram(image):
    """
    Equalizes the histogram of a grayscale uint8 image.

    Args:
        image (numpy.ndarray): Grayscale image with shape (height, width)
            and dtype uint8.

    Returns:
        numpy.ndarray: Equalized grayscale image with the same
            shape and dtype as the input.
    """
    result = image.copy()

    ### START CODE HERE ###

    # TODO

    ### END CODE HERE ###

    return result


def main():
    image = np.array(
        [
            [0, 0, 0, 1],
            [1, 1, 2, 2],
            [2, 3, 3, 3],
            [3, 3, 3, 3],
        ],
        dtype=np.uint8,
    )
    original = image.copy()
    expected = np.array(
        [
            [0, 0, 0, 59],
            [59, 59, 118, 118],
            [118, 255, 255, 255],
            [255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )

    result = equalize_histogram(image)

    assert np.array_equal(result, expected)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(image, original)
    assert result is not image

    constant = np.full((3, 3), 42, dtype=np.uint8)
    equalized_constant = equalize_histogram(constant)

    assert np.array_equal(equalized_constant, constant)
    assert equalized_constant is not constant

    print("Test passed!")


if __name__ == "__main__":
    main()
