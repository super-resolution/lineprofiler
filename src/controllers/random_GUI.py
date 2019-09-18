from viewer.random import Ui_MainWindow
from PyQt5 import QtWidgets,QtGui
from PyQt5.QtCore import QThread,pyqtSignal
import os
from controllers.interface import Interface
from controllers.image import ImageSIM




class MainWindow(Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()


    def init_component(self, qt_window):
        self.working_directory = os.getcwd()
        self.qt_window = qt_window
        self.plot_parameters = self.plotParameters.findChildren(QtGui.QCheckBox)

        self.interface = Interface(self)

        self._add_handlers()
        self.status_bar = QtWidgets.QStatusBar()
        self.qt_window.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Line Profiler ready to profile lines")


    def _add_handlers(self):



        self.pushButton_open.clicked.connect(
                lambda: self._open_files())
        self.pushButton_close_all.clicked.connect(
                lambda : self._close_all())
        self.pushButton_show.clicked.connect(
                lambda: self.interface.show_image(self.image_list.selectedItems()[0]))
        self.pushButton_close.clicked.connect(
            lambda: [self._close_image(i.row()) for i in self.image_list.selectedIndexes()]
        )
        self.pushButton_process.clicked.connect(
                lambda: (self.interface.start_thread(),
            self.status_bar.showMessage("Line Profiler profiling lines"),
            self.pushButton_process.setEnabled(False))
        )
        #self.spinBox_px_size.valueChanged.connect(self.interface.set_px_size)
        #self.spinBox_gaussian_blur.valueChanged.connect(self.interface.set_process_blur)
        self.comboBox_operation_mode.currentTextChanged.connect(self.interface.set_operation_mode)

        self.spinBox_lower_limit.valueChanged.connect(self.interface.set_process_lower_lim)
        self.spinBox_upper_limit.valueChanged.connect(self.interface.set_process_upper_lim)


        self.horizontalSlider_intensity_threshold.valueChanged.connect(
            lambda state, item=self.horizontalSlider_intensity_threshold,:
            self.doubleSpinBox_intensity_threshold.setValue(item.value()/10)
        )
        self.doubleSpinBox_intensity_threshold.valueChanged.connect(
            lambda state, item =self.doubleSpinBox_intensity_threshold :
            self.horizontalSlider_intensity_threshold.setValue(int(item.value()*10))

        )
        self.doubleSpinBox_expansion_factor.valueChanged.connect(
            self.interface.expansion_factor_changed
        )
        #self.doubleSpinBox_spline_parameter.valueChanged.connect(
        #    self.interface.spline_parameter_changed
        #)
        for i in (self.plot_parameters):
            i.stateChanged.connect(lambda: self.interface.checkbox_values_changed())
        for i in range(4):
            color = getattr(self, "comboBox_channel" + str(i) + "_color")
            channel = getattr(self, "checkBox_channel" + str(i))
            slider = getattr(self, "slider_channel" + str(i) + "_slice")
            getattr(self, "slider_channel" + str(i) + "_slice").setStyleSheet(
                "QSlider::sub-page:horizontal {background:" + color.currentText() + "}")

            color.currentIndexChanged.connect(
                lambda state, i=i, item=color: (
                    self.interface.set_channel_color(i, str(item.currentText())),
                    getattr(self, "slider_channel" + str(i) + "_slice").setStyleSheet(
                        "QSlider::sub-page:horizontal {background:" + item.currentText() + "}")))
            channel.stateChanged.connect(
                lambda state, i=i, item=channel: (
                    self.interface.set_channel_visible(i, item.isChecked()),
                    ))
            slider.valueChanged.connect(
                lambda state, i=i, item=slider,: self.interface.update_image(i, item.value()))

    def _increase_progress(self, value):
        self.progressBar.setValue(value)

    def _process_finished(self):
        self.status_bar.showMessage("Line Profiler ready to profile lines")
        self.pushButton_process.setEnabled(True)


    def _open_files(self):
        image = None
        file_dialog = QtWidgets.QFileDialog()
        title = "Open SIM files"
        # extensions = "Confocal images (*.jpg; *.png; *.tif;);;Confocal stacks (*.ics)"
        # extensions = "Confocal images (*.jpg *.png *.tif *.ics)"
        extensions = "image (*.czi *.tiff *.tif *.lsm *.png" \
                     ")"
        files_list = QtWidgets.QFileDialog.getOpenFileNames(file_dialog, title,
                                                            self.working_directory, extensions)[0]
        for file_ in files_list:
            image = ImageSIM(file_)
        if image is not None:
            self.image_list.addItem(image)

    def _close_all(self):
        image_list = self.image_list
        for i in range(image_list.count()):
            self._close_image(0)

    def _close_image(self, index):
        removed = self.image_list.takeItem(index)
        if removed:
            removed.reset_data()
        self.image_list.setCurrentRow(-1)