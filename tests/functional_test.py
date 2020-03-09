import unittest
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QStandardItemModel
from controllers.random_GUI import MainWindow
from controllers.image import *
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
        self.interface = self.mainWindow.interface


    def test_open_file_and_display_in_file_list(self):
        item = QListWidgetItem()
        self.mainWindow.image_list.addItem(item)
        row = ImageSIM(r"C:\Users\biophys\Pictures\Camera Roll\MAX_SIM3czi_Structured Illumination-CH1-CH2.tif")
        self.mainWindow.image_list.setItemWidget(item, row)
        item.setSizeHint(row.minimumSizeHint())
        self.mainWindow.image_list.item(0).setSelected(True)
        self.assertEqual(row, self.mainWindow.image_list.itemWidget(self.mainWindow.image_list.selectedItems()[0]))
        self.assertIsNotNone(row.data)
        self.interface.show_image(row)
        np.testing.assert_almost_equal(self.interface.current_image.data, row.data)
        print("Open file successfully tested")

    def test_run_multiple_files_in_different_modi_and_threads(self):
        image = ImageSIM(r"C:\Users\biophys\PycharmProjects\Fabi\data\test_data\MAX_3Farben-X1_16um_Out_Channel Alignment-5-X1.tif")
        self.interface.update_config()
        self.interface.config["intensity_threshold"] = 3
        self.interface.show_image(image)
        self.interface.start_thread()
        image = ImageSIM(r"C:\Users\biophys\PycharmProjects\Fabi\data\test_data_microtub\Expansion dSTORM-Line Profile test.tif")
        self.interface.update_config()
        self.interface.config["intensity_threshold"] = -3
        self.interface.show_image(image)
        self.interface.start_thread()
        self.assertIn("1",self.interface.processes)
        self.assertIn("2", self.interface.processes)

    def processing_configurations(self):
        self.interface.update_config()
        self.assertEqual(self.mainWindow.interface.config["blur"],
                         self.mainWindow.spinBox_gaussian_blur.value(), msg="Wrong blur value on thread initializdation")


        self.assertEqual(self.mainWindow.interface.config["px_size"],
                         self.mainWindow.spinBox_px_size.value(), msg="Wrong px_size on thread initializdation")


        self.assertEqual(self.mainWindow.interface.config["blur"],
                         self.mainWindow.spinBox_gaussian_blur.value(), msg="Wrong spline parameter on thread initializdation")


        self.assertEqual(self.mainWindow.interface.config["intensity_threshold"],
                         self.mainWindow.doubleSpinBox_intensity_threshold.value(), msg="Wrong intensity threshold on thread initializdation")

        self.mainWindow.spinBox_gaussian_blur.setValue(15)
        self.mainWindow.doubleSpinBox_spline_parameter.setValue(2.5)
        self.mainWindow.spinBox_px_size.setValue(2)
        self.mainWindow.doubleSpinBox_intensity_threshold.setValue(4)
        self.interface.update_config()

        self.assertEqual(self.mainWindow.interface.config["blur"], 15,
                         msg="Wrong blur value on update")
        self.assertEqual(self.mainWindow.interface.config["spline_parameter"], 2.5,
                         msg="Wrong spline parameter on update")
        self.assertEqual(self.mainWindow.interface.config["px_size"], 2,
                         msg="Wrong px_size on update")
        self.assertEqual(self.mainWindow.interface.config["intensity_threshold"],
                         4, msg="Wrong intensity threshold on update")

    def test_microtuboli_options(self):
        self.mainWindow.comboBox_operation_mode.setCurrentIndex(0)
        self.assertEqual(self.mainWindow.comboBox_operation_mode.currentText(), "Microtubule", msg="Wrong combo box Text")
        self.processing_configurations()
        print("Microtuboli Mode successfully tested")

    def test_snc_options(self):
        self.mainWindow.comboBox_operation_mode.setCurrentIndex(1)
        self.assertEqual(self.mainWindow.comboBox_operation_mode.currentText(), "SNC", msg="Wrong combo box Text")
        self.processing_configurations()
        print("SNC Mode successfully tested")


    def test_snc_one_channel(self):
        self.mainWindow.comboBox_operation_mode.setCurrentIndex(2)
        self.assertEqual(self.mainWindow.comboBox_operation_mode.currentText(), "SNC one channel", msg="Wrong combo box Text")
        self.processing_configurations()
        image = ImageSIM(r"C:\Users\biophys\PycharmProjects\Fabi\data\test_data\MAX_3Farben-X1_16um_Out_Channel Alignment-5-X1.tif")
        self.interface.update_config()
        self.interface.config["intensity_threshold"] = 3
        self.interface.show_image(image)
        self.interface.start_thread()
        print("SNC one channel Mode successfully tested")


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
        print("Plot options successfully tested")
        #self.end()

    def end(self):
        self.qtApp.exec_()

        self.qtApp.quit()
