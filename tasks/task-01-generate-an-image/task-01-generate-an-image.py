import numpy as np
import cv2


def generate_image(seed, width, height, mean, std):
    """
    Generates a grayscale image with pixel values sampled from a normal distribution.

    Args:
        seed (int): Random seed for reproducibility (student's registration number).
        width (int): Width of the generated image.
        height (int): Height of the generated image.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.

    Returns:
        image (numpy.ndarray): The generated image.
    """
    ### START CODE HERE ###
    ### TODO
    ### END CODE HERE ###
    return image


if __name__ == "__main__":

    # Test code
    seed = 0
    width = 3
    height = 5
    mean = 128
    std = 20

    image = generate_image(seed, width, height, mean, std)

    test = np.array(
        [
            [163.28104692, 136.00314417, 147.57475968],
            [172.81786398, 165.3511598, 108.4544424],
            [147.00176835, 124.97285583, 125.93562296],
            [136.21197004, 130.88087142, 157.08547014],
            [143.2207545, 130.43350033, 136.87726465],
        ]
    )

    assert (np.abs(image - test) < 0.0001).all()
    print("Test passed!")
