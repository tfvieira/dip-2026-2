import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class FourierTransformer:
    def __init__(self):
        pass

    def compute_fourier_transform(self, image):
        """Computes the Fourier Transform of an image."""
        dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(
            dft
        )  # Shift the zero-frequency component to the center
        return dft_shift

    def compute_inverse_fourier_transform(self, dft_shift):
        """Computes the inverse Fourier Transform from the shifted spectrum."""
        dft_ishift = np.fft.ifftshift(dft_shift)
        img_back = cv.idft(dft_ishift)
        img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        return img_back

    def visualize_fourier_transform(self, dft_shift):
        """Visualizes the magnitude and phase of the Fourier Transform."""
        magnitude_spectrum = 20 * np.log(
            cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1
        )
        phase_spectrum = np.arctan2(dft_shift[:, :, 1], dft_shift[:, :, 0])

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(magnitude_spectrum, cmap="gray")
        plt.title("Magnitude Spectrum")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(phase_spectrum, cmap="gray")
        plt.title("Phase Spectrum")
        plt.axis("off")

        plt.show()

    def generate_filter(
        self, shape, filter_type, cutoff_low=None, cutoff_high=None, n=2, ripple=0.1
    ):
        """
        Generates a low-pass, high-pass, band-pass, or band-reject filter.
        Supports Ideal, Butterworth, and Elliptic filters.
        """
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2  # Center coordinates
        mask = np.zeros((rows, cols), dtype=np.float32)

        for u in range(rows):
            for v in range(cols):
                distance = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)

                # Ideal Filters
                if filter_type == "ideal_low":
                    mask[u, v] = 1 if distance <= cutoff_low else 0
                elif filter_type == "ideal_high":
                    mask[u, v] = 1 if distance >= cutoff_low else 0
                elif filter_type == "ideal_band_pass":
                    mask[u, v] = 1 if cutoff_low <= distance <= cutoff_high else 0
                elif filter_type == "ideal_band_reject":
                    mask[u, v] = 1 if not (cutoff_low <= distance <= cutoff_high) else 0

                # Butterworth Filters
                elif filter_type == "butterworth_low":
                    mask[u, v] = 1 / (1 + (distance / cutoff_low) ** (2 * n))
                elif filter_type == "butterworth_high":
                    mask[u, v] = 1 / (1 + (cutoff_low / distance) ** (2 * n))
                elif filter_type == "butterworth_band_pass":
                    mask[u, v] = 1 / (
                        1
                        + (
                            (
                                (distance**2 - cutoff_low * cutoff_high)
                                / (distance * (cutoff_high - cutoff_low))
                            )
                            ** (2 * n)
                        )
                    )
                elif filter_type == "butterworth_band_reject":
                    mask[u, v] = 1 - mask[u, v]

                # Elliptic Filters (simplified approach for example purposes)
                elif filter_type == "elliptic_low":
                    mask[u, v] = 1 / (1 + ripple * (distance / cutoff_low) ** (2 * n))
                elif filter_type == "elliptic_high":
                    mask[u, v] = 1 / (1 + ripple * (cutoff_low / distance) ** (2 * n))

        return mask

    def apply_filter(self, dft_shift, mask):
        """Applies a filter to the Fourier Transform using Hadamard product."""
        filtered_dft = (
            dft_shift * mask[:, :, np.newaxis]
        )  # Extend mask to match dimensions
        return filtered_dft

    def create_notch_filter(self, shape, u, v, radius, filter_type="ideal"):
        """
        Creates a notch filter to eliminate impulses at specific locations.
        """
        mask = np.ones(shape[:2], dtype=np.float32)
        crow, ccol = shape[0] // 2, shape[1] // 2

        for du, dv in [(u, v), (-u, -v)]:
            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    distance = np.sqrt((x - crow - du) ** 2 + (y - ccol - dv) ** 2)
                    if filter_type == "ideal":
                        mask[x, y] = 0 if distance <= radius else mask[x, y]
                    elif filter_type == "butterworth":
                        mask[x, y] *= 1 / (1 + (distance / radius) ** 4)

        return mask
