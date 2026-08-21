import numpy as np


def correlate2d(image, kernel):
    """
    Applies 2D spatial correlation to a grayscale image.

    Args:
        image (numpy.ndarray): Grayscale image with shape (height, width).
        kernel (numpy.ndarray): 2D correlation kernel with odd dimensions.

    Returns:
        numpy.ndarray: Correlated image with the same shape as the input
            and dtype float64.
    """
    result = None

    ### START CODE HERE ###

    # TODO

    ### END CODE HERE ###

    return result


def main():
    image = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=np.uint8,
    )
    kernel = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [3, 5, 3],
            [9, 11, 6],
            [15, 17, 9],
        ],
        dtype=np.float64,
    )
    original_image = image.copy()
    original_kernel = kernel.copy()

    result = correlate2d(image, kernel)

    assert np.array_equal(result, expected)
    assert result.shape == image.shape
    assert result.dtype == np.float64
    assert np.array_equal(image, original_image)
    assert np.array_equal(kernel, original_kernel)
    assert result is not image

    identity_kernel = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.float64,
    )
    identity_result = correlate2d(image, identity_kernel)

    assert np.array_equal(identity_result, image.astype(np.float64))

    print("Test passed!")


if __name__ == "__main__":
    main()
