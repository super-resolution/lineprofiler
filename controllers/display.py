import pyqtgraph as pg
import numpy as np

class Display(object):
    def __init__(self, interface):
        self.widget = pg.PlotWidget()
        self.widget.setAspectLocked()
        self.interface = interface
        self.images = []
        for i in range(4):
            image = pg.ImageItem()
            self.images.append(image)
            self.widget.addItem(image)

    def show_image(self):
        images = self.interface.current_image.data_rgba_2d/self.interface.intensity_threshold
        images = np.clip(images, 0, 255)
        for i in range(images.shape[0]):
            self.images[i].setImage((images[i]).astype(np.uint8))
            self.images[i].show()