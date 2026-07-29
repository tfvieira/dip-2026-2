import matplotlib.pyplot as plt


class Visualization:
    def __init__(self):
        pass

    def plot_histogram(self, hist):
        plt.figure()
        plt.plot(hist)
        plt.title("Image Histogram")
        plt.show()
