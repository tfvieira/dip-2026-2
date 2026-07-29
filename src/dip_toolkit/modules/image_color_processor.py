import cv2
import matplotlib.pyplot as plt
import numpy as np


class ColorImageProcessor:
    def __init__(self):
        pass

    def plot_rgb_histograms(self, image):
        """Plots the histograms for the R, G, and B channels."""
        colors = ("b", "g", "r")
        plt.figure(figsize=(10, 5))
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        plt.title("RGB Histograms")
        plt.xlabel("Intensity Value")
        plt.ylabel("Count")
        plt.legend(["Blue", "Green", "Red"])
        plt.show()

    def convert_color_space(self, image, conversion):
        """Converts the image to the specified color space."""
        conversions = {
            "rgb_to_hsv": cv2.COLOR_BGR2HSV,
            "rgb_to_ycrcb": cv2.COLOR_BGR2YCrCb,
            "rgb_to_lab": cv2.COLOR_BGR2Lab,
            "rgb_to_gray": cv2.COLOR_BGR2GRAY,
        }
        if conversion not in conversions:
            supported = list(conversions.keys())

            raise ValueError(
                f"Unsupported conversion: {conversion}. Supported: {supported}"
            )
        return cv2.cvtColor(image, conversions[conversion])

    def rgb_to_cmyk(self, image):
        """Converts an RGB image to CMYK."""
        b, g, r = cv2.split(image)
        b, g, r = b / 255.0, g / 255.0, r / 255.0
        k = 1 - np.max([r, g, b], axis=0)
        c = (1 - r - k) / (1 - k + 1e-5)
        m = (1 - g - k) / (1 - k + 1e-5)
        y = (1 - b - k) / (1 - k + 1e-5)
        cmyk = np.stack((c, m, y, k), axis=-1)
        return (cmyk * 255).astype(np.uint8)

    def plot_rgb_3d_cube(self, image):
        """Plots a 3D scatter plot of RGB values on the RGB color cube."""
        r, g, b = cv2.split(image)
        r, g, b = r.flatten(), g.flatten(), b.flatten()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(r, g, b, c=image.reshape(-1, 3) / 255.0, marker=".")  # , alpha=0.1)
        ax.set_xlabel("Red Channel")
        ax.set_ylabel("Green Channel")
        ax.set_zlabel("Blue Channel")
        ax.set_title("RGB 3D Color Cube")
        plt.show()
