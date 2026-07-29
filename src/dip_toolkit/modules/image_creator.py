import numpy as np


class ImageCreator:
    def __init__(self):
        pass

    def create_filled_image(self, shape, value=0, dtype=np.uint8):
        """Creates an image filled with a specific scalar value."""
        return np.full(shape, value, dtype=dtype)

    def create_random_image(
        self, shape, distribution="uniform", dtype=np.uint8, **kwargs
    ):
        """Creates an image filled with random values based on a given distribution."""
        if distribution == "uniform":
            image = np.random.uniform(
                kwargs.get("low", 0), kwargs.get("high", 255), shape
            )
        elif distribution == "normal":
            image = np.random.normal(
                kwargs.get("mean", 128), kwargs.get("std", 20), shape
            )
        elif distribution == "rayleigh":
            image = np.random.rayleigh(kwargs.get("scale", 50), shape)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        # Clip values and cast to the desired dtype
        image = np.clip(
            image, 0, np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
        )
        return image.astype(dtype)

    def change_image_dtype(self, image, dtype):
        """Converts the data type of the given image."""
        return image.astype(dtype)

    def create_zeros_image(self, shape, dtype=np.uint8):
        """Creates an image filled with zeros."""
        return np.zeros(shape, dtype=dtype)

    def create_ones_image(self, shape, dtype=np.uint8):
        """Creates an image filled with ones."""
        return np.ones(shape, dtype=dtype) * (
            1 if np.issubdtype(dtype, np.floating) else 255
        )

    def create_salt_and_pepper_noise(
        self, height=100, width=100, salt_prob=0.05, pepper_prob=0.05
    ):
        """
        Returns an image ∈ [-1, 1] containing salt (I = 1.0) and
        pepper (I = -1.0) noise with respective probability distributions
        equal to salt_prob and pepper_prob. Pixels without noise have values of 0.5.
        """
        img = np.full((height, width), 0.5, dtype=np.float64)
        noise = np.random.rand(height, width)
        img[noise > 1 - salt_prob] = 1.0
        img[noise < pepper_prob] = -1.0
        return img
