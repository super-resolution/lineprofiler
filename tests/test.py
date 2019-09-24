from unittest import TestCase
import unittest
from matplotlib import cm

from src.controllers.fitter import *
from src.controllers.utility import *
from src.controllers.image import ImageSIM
from src.controllers import processing_SNC, processing_microtubule, processing_canny

import numpy as np
import cv2

class FitterTest(TestCase):
    def setUp(self):
        self.fitter = Fit()
        self.path = r"C:\Users\biophys\PycharmProjects\Fabi\data\test_data\MAX_3Farben-X1_16um_Out_Channel Alignment-5-X1.tif"
        image = ImageSIM(self.path)
        image.parse()
        self.data = image.data

    def test_fit_functions_setter(self):
        fit_function_name = {"gaussian", "bigaussian", "trigaussian", "cylinder_projection", "multi_cylinder_projection"}
        for func in fit_function_name:
            self.assertIn(func, fit_functions)
        fit_function_name.remove("multi_cylinder_projection")
        self.fitter.fit_function = fit_function_name
        fit_function_name_list = list(fit_function_name)
        fit_function_name_list.append("trigaussian")
        self.fitter.fit_function = fit_function_name_list
        self.fitter.fit_function = tuple(fit_function_name)
        self.assertEqual(len(fit_functions), len(fit_function_name))
        print(fit_functions)

    def test_fit_data(self):
        X = np.linspace(0, 200, 801)
        data = cylinder_projection.fit(X, 25, 100, 25/2+8.75, 25/2+8.75*2, 0, blur=38.73)
        self.fitter.fit_data(data, 400)

    def test_floodfill(self):
        image = np.zeros((1024,1024)).astype(np.uint8)
        image[512,:] = 255
        image[520,:] = 255
        image[512:520,0] = 255
        image[512:520,1023] = 255

        image = create_floodfill_image(image)
        self.assertEqual(len(np.where(image != 0)[0]), 7154)#1022*7 pixel != 0

    def test_processing_SNC(self):
        thread = processing_SNC.QProcessThread(cm.gist_ncar)
        thread.set_data(self.data, self.path)
        thread.blur = 20
        thread.px_size = 0.032
        thread.intensity_threshold = 3.9
        thread.run()

    def test_processing_canny(self):
        thread = processing_canny.QProcessThread(cm.gist_ncar)
        thread.set_data(self.data, self.path)
        thread.blur = 9
        thread.px_size = 0.032
        thread.intensity_threshold = 3.9
        thread.run()

    def test_processing_microtuboli(self):
        thread = processing_microtubule.QProcessThread(cm.gist_ncar)
        thread.set_data(self.data, self.path)
        thread.blur = 20
        thread.px_size = 0.032
        thread.intensity_threshold = 3.9
        thread.run()



if __name__ == '__main__':
    unittest.main()