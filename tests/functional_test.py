import unittest
from PyQt5.QtWidgets import QApplication, QMainWindow
from controllers.random_GUI import MainWindow
import sys
import numpy as np

class FunctionalTestLineProfiler(unittest.TestCase):
    def setUp(self):
        self.qtApp = QApplication(sys.argv)
        qtWindow = QMainWindow()
        self.mainWindow = MainWindow()
        self.mainWindow.setupUi(qtWindow)

        self.mainWindow.init_component(qtWindow)
        qtWindow.show()
    def test_open_file(self):
        pass

    def processing_configurations(self):
        self.assertEqual(self.mainWindow.interface.current_processing_thread.blur,
                         self.mainWindow.spinBox_gaussian_blur.value(), msg="Wrong blur value on thread initializdation")
        self.mainWindow.spinBox_gaussian_blur.setValue(15)
        self.assertEqual(self.mainWindow.interface.current_processing_thread.blur, 15,
                         msg="Wrong blur value on update")

        self.assertEqual(self.mainWindow.interface.current_processing_thread.px_size,
                         self.mainWindow.spinBox_px_size.value(), msg="Wrong px_size on thread initializdation")
        self.mainWindow.spinBox_px_size.setValue(2)
        self.assertEqual(self.mainWindow.interface.current_processing_thread.px_size, 2,
                         msg="Wrong px_size on update")

        self.assertEqual(self.mainWindow.interface.current_processing_thread.blur,
                         self.mainWindow.spinBox_gaussian_blur.value(), msg="Wrong spline parameter on thread initializdation")
        self.mainWindow.doubleSpinBox_spline_parameter.setValue(2.5)
        self.assertEqual(self.mainWindow.interface.current_processing_thread.spline_parameter, 2.5,
                         msg="Wrong spline parameter on update")

        self.assertEqual(self.mainWindow.interface.current_processing_thread.intensity_threshold,
                         np.exp(self.mainWindow.doubleSpinBox_intensity_threshold.value()), msg="Wrong intensity threshold on thread initializdation")
        self.mainWindow.doubleSpinBox_intensity_threshold.setValue(4)
        self.assertEqual(self.mainWindow.interface.current_processing_thread.intensity_threshold,
                         np.exp(4), msg="Wrong intensity threshold on update")

    def test_microtuboli_options(self):
        self.mainWindow.comboBox_operation_mode.setCurrentIndex(0)
        self.assertEqual(self.mainWindow.comboBox_operation_mode.currentText(), "Microtuboli", msg="Wrong combo box Text")
        self.processing_configurations()

    def test_snc_options(self):
        self.mainWindow.comboBox_operation_mode.setCurrentIndex(1)
        self.assertEqual(self.mainWindow.comboBox_operation_mode.currentText(), "SNC", msg="Wrong combo box Text")
        self.processing_configurations()

    def test_snc_one_channel(self):
        self.mainWindow.comboBox_operation_mode.setCurrentIndex(2)
        self.assertEqual(self.mainWindow.comboBox_operation_mode.currentText(), "SNC one channel", msg="Wrong combo box Text")
        self.processing_configurations()

    def test_plot_options(self):
        self.mainWindow.comboBox_operation_mode.setCurrentIndex(0)
        self.mainWindow.checkBox_bigaussian.setChecked(True)
        self.mainWindow.checkBox_trigaussian.setChecked(True)
        self.mainWindow.checkBox_cylinder_projection.setChecked(True)
        self.mainWindow.checkBox_multi_cylidner_projection.setChecked(True)
        self.mainWindow.checkBox_gaussian.setChecked(True)
        functions = []
        for box in self.mainWindow.plot_parameters:
            if box.isChecked():
                functions.append(box.text().lower())
        self.assertEqual(self.mainWindow.interface.fitter.fit_functions, functions)
        self.tearDown()


    def tearDown(self):
        self.qtApp.exec_()