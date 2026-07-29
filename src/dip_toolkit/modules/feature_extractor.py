import cv2 as cv


class FeatureExtractor:
    def __init__(self):
        pass

    def extract_edges(self, image):
        edges = cv.Canny(image, 100, 200)
        return edges
