import cv2 as cv
import numpy as np


class ImageTransformer:
    def __init__(self):
        pass

    def translate(self, image, tx, ty):
        """Translates the image by tx (x-axis) and ty (y-axis)."""
        rows, cols = image.shape[:2]
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv.warpAffine(image, translation_matrix, (cols, rows))

    def rotate(self, image, angle, center=None, scale=1.0):
        """Rotates the image around a center point."""
        rows, cols = image.shape[:2]
        if center is None:
            center = (cols // 2, rows // 2)
        rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
        return cv.warpAffine(image, rotation_matrix, (cols, rows))

    def mirror(self, image, axis="x"):
        """Mirrors the image along the specified axis (x, y, or both)."""
        if axis == "x":
            return cv.flip(image, 0)  # Flip vertically
        elif axis == "y":
            return cv.flip(image, 1)  # Flip horizontally
        elif axis == "xy":
            return cv.flip(image, -1)  # Flip both vertically and horizontally
        else:
            raise ValueError("Invalid axis. Choose 'x', 'y', or 'xy'.")

    def stretch(self, image, fx, fy):
        """Stretches (scales) the image along x and y axes."""
        return cv.resize(image, None, fx=fx, fy=fy, interpolation=cv.INTER_LINEAR)

    def crop(self, image, x_start, y_start, width, height):
        """Crops the image to the specified rectangle."""
        return image[y_start : y_start + height, x_start : x_start + width]

    def resize(self, image, width, height):
        """Resizes the image to the specified dimensions."""
        return cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)
