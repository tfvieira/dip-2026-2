import cv2 as cv


class ImageDrawer:
    def __init__(self):
        pass

    def draw_circle(self, image, center, radius, color, thickness=2):
        """Draws a circle on the image."""
        return cv.circle(image.copy(), center, radius, color, thickness)

    def draw_line(self, image, start_point, end_point, color, thickness=2):
        """Draws a line on the image."""
        return cv.line(image.copy(), start_point, end_point, color, thickness)

    def draw_rectangle(self, image, top_left, bottom_right, color, thickness=2):
        """Draws a rectangle on the image."""
        return cv.rectangle(image.copy(), top_left, bottom_right, color, thickness)

    def draw_ellipse(
        self, image, center, axes, angle, start_angle, end_angle, color, thickness=2
    ):
        """Draws an ellipse on the image."""
        return cv.ellipse(
            image.copy(), center, axes, angle, start_angle, end_angle, color, thickness
        )

    def put_text(
        self,
        image,
        text,
        position,
        font=cv.FONT_HERSHEY_SIMPLEX,
        font_scale=1,
        color=(255, 255, 255),
        thickness=2,
    ):
        """Draws text on the image."""
        return cv.putText(
            image.copy(), text, position, font, font_scale, color, thickness
        )

    def mask_image(self, image, mask):
        """
        Masks the input image with another image (mask).
        Both images must have the same dimensions.
        """
        if image.shape != mask.shape:
            raise ValueError("Image and mask must have the same dimensions.")
        return cv.bitwise_and(image, mask)
