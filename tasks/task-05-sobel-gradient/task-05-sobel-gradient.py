import numpy as np


def sobel_gradient(image):
    """
    Computes the Sobel gradient of a grayscale image.

    Args:
        image (numpy.ndarray): Grayscale image with shape (height, width).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            Sobel gradient in X, Sobel gradient in Y, and gradient magnitude.
    """
    gradient_x = None
    gradient_y = None
    magnitude = None

    ### START CODE HERE ###

    # TODO

    ### END CODE HERE ###

    return gradient_x, gradient_y, magnitude


def main():
    image = np.array(
        [
            [0, 0, 255],
            [0, 0, 255],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )
    expected_x = np.array(
        [
            [0, 765, 0],
            [0, 1020, 0],
            [0, 765, 0],
        ],
        dtype=np.float64,
    )
    expected_y = np.array(
        [
            [0, 255, 510],
            [0, 0, 0],
            [0, -255, -510],
        ],
        dtype=np.float64,
    )
    expected_magnitude = np.sqrt(expected_x**2 + expected_y**2)
    original_image = image.copy()

    gradient_x, gradient_y, magnitude = sobel_gradient(image)

    assert np.array_equal(gradient_x, expected_x)
    assert np.array_equal(gradient_y, expected_y)
    assert np.allclose(magnitude, expected_magnitude)
    assert gradient_x.shape == image.shape
    assert gradient_y.shape == image.shape
    assert magnitude.shape == image.shape
    assert gradient_x.dtype == np.float64
    assert gradient_y.dtype == np.float64
    assert magnitude.dtype == np.float64
    assert np.array_equal(image, original_image)
    assert gradient_x is not image
    assert gradient_y is not image
    assert magnitude is not image

    zero_image = np.zeros((3, 3), dtype=np.uint8)
    expected_zero = np.zeros((3, 3), dtype=np.float64)
    zero_x, zero_y, zero_magnitude = sobel_gradient(zero_image)

    assert np.array_equal(zero_x, expected_zero)
    assert np.array_equal(zero_y, expected_zero)
    assert np.array_equal(zero_magnitude, expected_zero)

    print("Test passed!")


if __name__ == "__main__":
    main()
