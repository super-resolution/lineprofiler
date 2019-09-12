
from controllers.display import *
from controllers import processing_SNC
from controllers import processing_green
from controllers import processing_canny

from controllers.utility import *
from controllers.fitter import *
from PyQt5 import QtCore

class Interface():
    def __init__(self, main_window):
        self.current_image = None
        self.main_window = main_window
        self.display = Display(self)
        self.main_window.viewer_container_layout.addWidget(self.display.widget)
        self.intensity_threshold = 30
        self.pixel_size = self.main_window.spinBox_px_size.value()
        self.fitter = Fit()
        self.current_spline = None
        self.checkbox_values_changed()

        self.set_operation_mode(self.main_window.comboBox_operation_mode.currentText())
        self.main_window.horizontalSlider_intensity_threshold.valueChanged.connect(
            lambda v: (setattr(self.display, "intensity_threshold", v / 10),(self.slider_threshold_changed())))



    def start_thread(self):
        self.current_processing_thread.start()

    def fit_data(self, data, center, i, path, color, size):
        self.fitter.fit_data(data, center=center, nth_line=i, path=path,
                             c=color, n_profiles=size)

    def set_channel_color(self):
        pass

    def set_handlers(self):
        self.main_window.spinBox_px_size.valueChanged.connect(
            lambda v: setattr(self.current_processing_thread, "px_size", v))
        self.main_window.spinBox_gaussian_blur.valueChanged.connect(
            lambda v: setattr(self.current_processing_thread, "blur", v))
        self.main_window.horizontalSlider_intensity_threshold.valueChanged.connect(
            lambda v: setattr(self.current_processing_thread, "intensity_threshold", v/10))

        self.main_window.doubleSpinBox_spline_parameter.valueChanged.connect(
            lambda v: setattr(self.current_processing_thread, "spline_parameter", v))


        self.current_processing_thread.px_size = self.main_window.spinBox_px_size.value()
        self.current_processing_thread.blur = self.main_window.spinBox_gaussian_blur.value()
        self.current_processing_thread.intensity_threshold = self.main_window.horizontalSlider_intensity_threshold.value()/10
        self.current_processing_thread.spline_parameter = self.main_window.doubleSpinBox_spline_parameter.value()


    #@QtCore.pyqtSlot(float)
    def set_operation_mode(self, value):
        if value == "Microtuboli":
            self.current_processing_thread = processing_green.QProcessThread()

            self.main_window.checkBox_multi_cylidner_projection.setEnabled(True)
            self.main_window.checkBox_multi_cylidner_projection.setChecked(True)

            self.main_window.checkBox_cylinder_projection.setEnabled(True)
            self.main_window.checkBox_cylinder_projection.setChecked(True)

            self.main_window.doubleSpinBox_expansion_factor.setEnabled(True)
            self.main_window.spinBox_lower_limit.setEnabled(False)
            self.main_window.spinBox_upper_limit.setEnabled(False)
            self.main_window.spinBox_px_size.setValue(0.01)
            self.main_window.spinBox_gaussian_blur.setValue(20)


        elif value == "SNC":
            self.current_processing_thread = processing_SNC.QProcessThread()
            self.main_window.checkBox_multi_cylidner_projection.setEnabled(False)
            self.main_window.checkBox_multi_cylidner_projection.setChecked(False)

            self.main_window.checkBox_cylinder_projection.setEnabled(False)
            self.main_window.checkBox_cylinder_projection.setChecked(False)

            self.main_window.doubleSpinBox_expansion_factor.setEnabled(False)
            self.main_window.spinBox_lower_limit.setEnabled(True)
            self.main_window.spinBox_upper_limit.setEnabled(True)
            self.main_window.spinBox_px_size.setValue(0.032)
            self.main_window.spinBox_gaussian_blur.setValue(20)

        elif value == "SNC one channel":
            self.current_processing_thread = processing_canny.QProcessThread()

            self.main_window.checkBox_multi_cylidner_projection.setEnabled(False)
            self.main_window.checkBox_multi_cylidner_projection.setChecked(False)

            self.main_window.checkBox_cylinder_projection.setEnabled(False)
            self.main_window.checkBox_cylinder_projection.setChecked(False)

            self.main_window.doubleSpinBox_expansion_factor.setEnabled(False)
            self.main_window.spinBox_lower_limit.setEnabled(True)
            self.main_window.spinBox_upper_limit.setEnabled(True)
            self.main_window.spinBox_px_size.setValue(0.032)
            self.main_window.spinBox_gaussian_blur.setValue(9)


        self.set_handlers()
        self.current_processing_thread.sig_plot_data.connect(self.fit_data)
        self.current_processing_thread.sig.connect(self.main_window._increase_progress)
        self.current_processing_thread.done.connect(self.main_window._process_finished)
        self.checkbox_values_changed()
        if self.current_image is not None:
            self.push_image_to_thread()



    #@QtCore.pyqtSlot(int)
    #def set_process_blur(self, value):
    #    self.current_processing_thread.blur = value

    #@QtCore.pyqtSlot(int)
    def set_process_lower_lim(self, value):
        self.current_processing_thread.lower_lim = value

    #@QtCore.pyqtSlot(int)
    def set_process_upper_lim(self, value):
        self.current_processing_thread.upper_lim = value

    #def set_px_size(self, value):
        #self.pixel_size = value
        #self.current_processing_thread.pixel_size = value

    def set_channel_visible(self, i, enabled):
        self.current_image.channel = (i,enabled)
        self.display.show_image()

    def slider_changed(self, ch, i):
        self.current_image.index = ch,i

    def slider_threshold_changed(self):
        #self.current_processing_thread.intensity_threshold = th
        #self.display.intensity_threshold = th
        try:
            self.display.show_image()
        except:
            ValueError("No image")

    #def spline_parameter_changed(self, value):
    #    self.current_processing_thread.spline_parameter = value

    def expansion_factor_changed(self, value):
        self.fitter.expansion = value

    def checkbox_values_changed(self):
        functions = []
        for box in self.main_window.plot_parameters:
            if box.isChecked():
                functions.append(box.text().lower())
        if len(functions)==0:
            raise ValueError("Select minimum one checkbox")
        self.fitter.fit_functions = functions

    def push_image_to_thread(self):
        #px_size = self.main_window.spinBox_px_size.value()
        self.current_processing_thread.set_data(self.current_image.data, self.current_image.file_path)

    def show_image(self, image):
        if image[0].isParsingNeeded:
            image[0].parse(calibration_px=self.pixel_size)
        self.current_image = image[0]
        for i in range(self.current_image.metaData["ShapeSizeC"]):
            box = getattr(self.main_window, "checkBox_channel" + str(i))
            box.setEnabled(True)
            slider = getattr(self.main_window, "slider_channel" + str(i) + "_slice")
            slider.setMaximum(self.current_image.metaData["ShapeSizeZ"]-1)
        print(self.main_window.checkBox_channel0.isCheckable())
        self.push_image_to_thread()

    def update_image(self, channel, value):
        self.current_image.index = (channel, value)
        self.display.show_image()
