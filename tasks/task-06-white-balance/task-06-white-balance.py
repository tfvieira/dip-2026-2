import numpy as np


def white_balance(image):
    """
    Applies Gray World white balance to an RGB image.

    Args:
        image (numpy.ndarray): RGB image with shape (height, width, 3) and
            dtype uint8.

    Returns:
        numpy.ndarray: White-balanced RGB image with the same shape and dtype
            as the input.
    """
    result = None

    ### START CODE HERE ###

    # TODO

    ### END CODE HERE ###

    return result


def main():
    image = np.array(
        [
            [[20, 50, 120], [40, 100, 150]],
            [[60, 150, 180], [80, 100, 150]],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [[40, 50, 80], [80, 100, 100]],
            [[120, 150, 120], [160, 100, 100]],
        ],
        dtype=np.uint8,
    )
    original = image.copy()

    result = white_balance(image)

    assert np.array_equal(result, expected)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(image, original)
    assert result is not image
    assert np.all(result >= 0)
    assert np.all(result <= 255)

    neutral = np.array(
        [
            [[50, 50, 50], [100, 100, 100]],
            [[150, 150, 150], [200, 200, 200]],
        ],
        dtype=np.uint8,
    )
    balanced_neutral = white_balance(neutral)
    assert np.array_equal(balanced_neutral, neutral)
    assert balanced_neutral is not neutral

    zero_channel = np.array(
        [
            [[0, 60, 120], [0, 120, 180]],
            [[0, 60, 120], [0, 120, 180]],
        ],
        dtype=np.uint8,
    )
    balanced_zero = white_balance(zero_channel)
    assert np.all(balanced_zero[:, :, 0] == 0)
    assert balanced_zero.dtype == np.uint8
    assert balanced_zero.shape == zero_channel.shape

    print("Test passed!")


if __name__ == "__main__":
    main()
