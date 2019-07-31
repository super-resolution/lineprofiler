import pyqtgraph as pg
import numpy as np

class Display(object):
    def __init__(self, interface):
        self.widget = pg.PlotWidget()
        self.widget.setAspectLocked()
        self.interface = interface
        self._intensity_threshold = 1
        self.images = []
        #self.line_plot = pg.PlotItem()
        #self.widget.addItem(self.line_plot)
        for i in range(4):
            image = pg.ImageItem()
            self.images.append(image)
            self.widget.addItem(image)

    def show_image(self):
        images = self.interface.current_image.data_rgba_2d/self._intensity_threshold
        images = np.clip(images, 0, 255)
        for i in range(images.shape[0]):
            self.images[i].setImage((images[i]).astype(np.uint8))
            self.images[i].show()

    def plot_line(self):#todo: show spline before run
        line = self.interface.current_spline
        if line is not None:
            self.line_plot.clear()
            self.line_plot.plot()

    @property
    def intensity_threshold(self):
        return self._intensity_threshold

    @intensity_threshold.setter
    def intensity_threshold(self, value):
        self._intensity_threshold = np.exp(value)