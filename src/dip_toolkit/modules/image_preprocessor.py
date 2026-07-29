import cv2 as cv


class ImagePreprocessor:
    def __init__(self):
        pass

    def resize_image(self, image, width, height):
        return cv.resize(image, (width, height))

    def apply_filter(self, image, filter_type="blur", **kwargs):
        ksize = kwargs.get("ksize", 5)

        if filter_type == "blur":
            return cv.GaussianBlur(image, (ksize, ksize), 0)

        elif filter_type == "median":
            # ksize must be odd and greater than 1
            if ksize % 2 == 0:
                ksize += 1
            return cv.medianBlur(image, ksize)

        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
