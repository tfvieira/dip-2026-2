import cv2 as cv


class ImageAnalysis:
    def __init__(self):
        pass

    def compute_histogram(self, image):
        hist = cv.calcHist([image], [0], None, [256], [0, 256])
        return hist
