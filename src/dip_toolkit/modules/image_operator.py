import cv2


class ImageOperator:
    def __init__(self):
        pass

    def add_images(self, image1, image2):
        """
        Adds two images pixel-by-pixel.
        """
        return cv2.add(image1, image2)

    def subtract_images(self, image1, image2):
        """
        Subtracts the second image from the first image pixel-by-pixel.
        """
        return cv2.subtract(image1, image2)

    def multiply_images(self, image1, image2):
        """
        Multiplies two images pixel-by-pixel.
        """
        return cv2.multiply(image1, image2)

    def divide_images(self, image1, image2):
        """
        Divides the first image by the second image pixel-by-pixel.
        """
        return cv2.divide(image1, image2)

    def bitwise_multiply_images(self, image1, image2):
        """
        Performs bitwise pixel multiplication between two images.
        """
        return cv2.bitwise_and(image1, image2)
