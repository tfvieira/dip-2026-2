import cv2


class WebcamStreamReader:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open webcam with index {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_frame(self):
        if self.cap is None:
            raise RuntimeError("Camera is not opened. Call open() first.")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from webcam.")
        return frame

    def show_stream(self, window_name="Webcam Stream", exit_key="q"):
        if self.cap is None:
            self.open()
        print(f"Press '{exit_key}' to exit.")
        while True:
            frame = self.read_frame()
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord(exit_key):
                break
        self.close()

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
