from controllers.display import *
from controllers.fitter import Fit, Hist
from matplotlib import cm
import weakref
from controllers.factory import process_factory



class Interface():
    def __init__(self, main_window):
        self.current_image = None
        self.main_window = main_window
        self.display = Display(self)
        self.main_window.viewer_container_layout.addWidget(self.display.widget)
        self.intensity_threshold = 30
        self.pixel_size = self.main_window.spinBox_px_size.value()
        self.current_spline = None
        self.fitter = Fit()
        self.checkbox_values_changed()
        self.set_operation_mode(self.main_window.comboBox_operation_mode.currentText())
        self.config = {"intensity_threshold":0,
                       "px_size":0,
                       "blur":0,
                       "spline_parameter":0,
                       "distance_threshold":0,
                       "upper_limit":0,
                       "lower_limit":0,
                       "profil_width":0}

        self.main_window.horizontalSlider_intensity_threshold.valueChanged.connect(
            lambda v: (setattr(self.display, "intensity_threshold", v / 10),(self.slider_threshold_changed())))
        self.processes = dict()
        self.processID = 0

    def del_process(self, id):
        process,progress_bar = self.processes.pop(str(id))
        self.main_window.annihilate_progress_bar(progress_bar)

    def start_thread(self):
        self.update_config()
        self.processID += 1
        progress_bar = self.main_window.create_progress_bar()
        progress_bar.setFormat(self.current_image.file_path.split("/")[-1])
        process = process_factory(self.main_window.comboBox_operation_mode.currentText(), self.current_image,
                                  self.config, self.fit_data, self.del_process, progress_bar, self.processID)
        self.processes[str(self.processID)] = (process, progress_bar)
        process.start()

    def fit_data(self, data, center, i, path, color, size):
        self.fitter.fit_data(data, center=center, nth_line=i, path=path,
                             c=color, n_profiles=size)

    def set_channel_color(self):
        pass

    def update_config(self):
        self.config["intensity_threshold"] = self.main_window.horizontalSlider_intensity_threshold.value()/10
        self.config["px_size"] = self.main_window.spinBox_px_size.value()
        self.config["spline_parameter"] = self.main_window.doubleSpinBox_spline_parameter.value()
        self.config["blur"] = self.main_window.spinBox_gaussian_blur.value()
        self.config["distance_threshold"] = self.main_window.spinBox_lower_limit.value()
        self.config["upper_limit"] = self.main_window.spinBox_upper_limit.value()
        self.config["lower_limit"] = self.main_window.spinBox_lower_limit.value()
        self.config["profil_width"] = self.main_window.spinBox_profil_width.value()

    # def set_handlers(self):
    #     self.main_window.spinBox_px_size.valueChanged.connect(
    #         lambda v: setattr(self.current_processing_thread, "px_size", v))
    #     self.main_window.spinBox_gaussian_blur.valueChanged.connect(
    #         lambda v: setattr(self.current_processing_thread, "blur", v))
    #     self.main_window.horizontalSlider_intensity_threshold.valueChanged.connect(
    #         lambda v: setattr(self.current_processing_thread, "intensity_threshold", v/10))
    #
    #     self.main_window.doubleSpinBox_spline_parameter.valueChanged.connect(
    #         lambda v: setattr(self.current_processing_thread, "spline_parameter", v))
    #
    #
    #     self.current_processing_thread.px_size = self.main_window.spinBox_px_size.value()
    #     self.current_processing_thread.blur = self.main_window.spinBox_gaussian_blur.value()
    #     self.current_processing_thread.intensity_threshold = self.main_window.horizontalSlider_intensity_threshold.value()/10
    #     self.current_processing_thread.spline_parameter = self.main_window.doubleSpinBox_spline_parameter.value()


    #@QtCore.pyqtSlot(float)
    def set_operation_mode(self, value):
        if value == "Microtubule":

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

            self.main_window.checkBox_multi_cylidner_projection.setEnabled(False)
            self.main_window.checkBox_multi_cylidner_projection.setChecked(False)

            self.main_window.checkBox_cylinder_projection.setEnabled(False)
            self.main_window.checkBox_cylinder_projection.setChecked(False)

            self.main_window.doubleSpinBox_expansion_factor.setEnabled(False)
            self.main_window.spinBox_lower_limit.setEnabled(True)
            self.main_window.spinBox_upper_limit.setEnabled(True)
            self.main_window.spinBox_px_size.setValue(0.032)
            self.main_window.spinBox_gaussian_blur.setValue(9)
    #
    #
    #     self.set_handlers()
    #     self.current_processing_thread.sig_plot_data.connect(self.fit_data)
    #     self.current_processing_thread.sig.connect(self.main_window._increase_progress)
    #     self.current_processing_thread.done.connect(self.process_finished)
    #     self.checkbox_values_changed()
    #     if self.current_image is not None:
    #         self.push_image_to_thread()


    def process_finished(self):
        self.main_window._process_finished()


    # def set_process_lower_lim(self, value):
    #     self.current_processing_thread.lower_lim = value
    #
    # def set_process_upper_lim(self, value):
    #     self.current_processing_thread.upper_lim = value

    def set_channel_visible(self, i, enabled):
        self.current_image.channel = (i, enabled)
        self.display.show_image()

    def slider_changed(self, ch, i):
        self.current_image.index = ch,i

    def slider_threshold_changed(self):
        try:
            self.display.show_image()
        except:
            ValueError("No image")

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

    # def push_image_to_thread(self):
    #     self.current_processing_thread.set_data(self.current_image.data, self.current_image.file_path)

    def show_image(self, image):
        if image.isParsingNeeded:
            image.parse(calibration_px=self.pixel_size)
        self.current_image = image
        for i in range(self.current_image.metaData["ShapeSizeC"]):
            box = getattr(self.main_window, "checkBox_channel" + str(i))
            box.setEnabled(True)
            slider = getattr(self.main_window, "slider_channel" + str(i) + "_slice")
            slider.setMaximum(self.current_image.metaData["ShapeSizeZ"]-1)
        print(self.main_window.checkBox_channel0.isCheckable())
        self.display.show_image()
        #self.push_image_to_thread()

    def plot_histogram(self):
        l = []
        for i in range(self.main_window.image_list.count()):
            l.append(self.main_window.image_list.itemWidget(self.main_window.image_list.item(i)))
        histogram = Hist(l)
        histogram.create_histogram()

    def update_image(self, channel, value):
        self.current_image.index = (channel, value)
        self.display.show_image()
