
from controllers.display import *
from controllers.processing_green import *
from PyQt5 import QtCore

class interface():
    def __init__(self, main_window):
        self.current_image = []
        self.main_window = main_window
        self.display = Display(self)
        self.main_window.viewer_container_layout.addWidget(self.display.widget)
        self.current_processing_thread = QProcessThread()
        self.intensity_threshold = 30

    def set_channel_color(self):
        a=0

    #@QtCore.pyqtSlot(float)
    def set_process_distance_th(self, value):
        self.current_processing_thread.distance_transform_th = value

    #@QtCore.pyqtSlot(int)
    def set_process_blur(self, value):
        self.current_processing_thread.blur = value

    #@QtCore.pyqtSlot(int)
    def set_process_lower_lim(self, value):
        self.current_processing_thread.lower_lim = value

    #@QtCore.pyqtSlot(int)
    def set_process_upper_lim(self, value):
        self.current_processing_thread.upper_lim = value

    def set_channel_visible(self, i, enabled):
        self.current_image.channel = (i,enabled)
        self.display.show_image()

    def slider_changed(self, ch, i):
        self.current_image.index = ch,i

    def slider_threshold_changed(self, th):
        self.current_processing_thread.intensity_threshold = th
        self.intensity_threshold = th
        try:
            self.display.show_image()
        except:
            ValueError("No image")

    def show_image(self, image):
        if image[0].isParsingNeeded:
            image[0].parse()
        self.current_image = image[0]
        for i in range(self.current_image.metaData["ShapeSizeC"]):
            box = getattr(self.main_window, "checkBox_channel" + str(i))
            box.setEnabled(True)
            slider = getattr(self.main_window, "slider_channel" + str(i) + "_slice")
            slider.setMaximum(self.current_image.metaData["ShapeSizeZ"]-1)
        print(self.main_window.checkBox_channel0.isCheckable())
        self.current_processing_thread.set_data(self.current_image.data, self.current_image.metaData["SizeX"])

    def update_image(self, channel, value):
        self.current_image.index = (channel, value)
        self.display.show_image()
