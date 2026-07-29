import matplotlib.pyplot as plt
import pywt


class WaveletTransformer:
    def __init__(self):
        pass

    def wavelet_decompose(self, image, wavelet="haar", level=1):
        """
        Decomposes an image using discrete wavelet transform (DWT).
        Returns approximation coefficients and detailed coefficients.
        """
        coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
        return coeffs

    def wavelet_reconstruct(self, coeffs, wavelet="haar"):
        """
        Reconstructs an image from wavelet coefficients using inverse DWT.
        """
        return pywt.waverec2(coeffs, wavelet=wavelet)

    def visualize_coeffs(self, coeffs):
        """
        Visualizes approximation and detail coefficients for one level.
        """
        approximation, (horizontal, vertical, diagonal) = coeffs

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(approximation, cmap="gray")
        plt.title("Approximation Coefficients")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(horizontal, cmap="gray")
        plt.title("Horizontal Detail Coefficients")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(vertical, cmap="gray")
        plt.title("Vertical Detail Coefficients")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(diagonal, cmap="gray")
        plt.title("Diagonal Detail Coefficients")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def threshold_coeffs(self, coeffs, threshold):
        """
        Applies a soft threshold to the detailed coefficients.
        """
        thresholded_coeffs = [coeffs[0]]  # Approximation coefficients remain unchanged
        for detail in coeffs[1:]:
            thresholded_detail = tuple(
                pywt.threshold(c, threshold, mode="soft") for c in detail
            )
            thresholded_coeffs.append(thresholded_detail)
        return thresholded_coeffs
