from unittest import TestCase
import unittest
from matplotlib import cm

from src.controllers.fitter import *
from src.controllers.utility import *
from src.controllers.image import ImageSIM
from src.controllers import processing_SNC, processing_microtubule, processing_canny
from src.controllers.profile_handler import profile_collector, profile_painter

import numpy as np
import os
import cv2

class FitterTest(TestCase):
    def setUp(self):
        self.fitter = Fit()
        self.path = r"C:\Users\biophys\PycharmProjects\Fabi\data\test_data\MAX_3Farben-X1_16um_Out_Channel Alignment-5-X1.tif"
        image = ImageSIM(self.path)
        image.parse()
        self.data = image.data
        self.save_path = os.path.dirname(os.getcwd()) + "\\data\\" + os.path.splitext(os.path.basename(self.path))[0]
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

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

    def painter(self):
        self.assertTrue(os.path.exists(self.save_path+r"\Image_with_profiles.tif"))
        self.assertTrue(os.path.exists(self.save_path+r'\Image_overlay.tif'))
        os.remove(self.save_path + r"\Image_with_profiles.tif")
        os.remove(self.save_path + r'\Image_overlay.tif')

    def test_processing_SNC(self):
        thread = processing_SNC.QProcessThread(cm.gist_ncar)
        thread.set_data(self.data, self.path)
        thread.blur = 20
        thread.px_size = 0.032
        thread.intensity_threshold = 3.9
        thread.run()
        self.painter()


    def test_processing_canny(self):
        thread = processing_canny.QProcessThread(cm.gist_ncar)
        thread.set_data(self.data, self.path)
        thread.blur = 9
        thread.px_size = 0.032
        thread.intensity_threshold = 3.9
        thread.run()
        self.painter()

    def test_processing_microtuboli(self):
        thread = processing_microtubule.QProcessThread(cm.gist_ncar)
        thread.set_data(self.data, self.path)
        thread.blur = 20
        thread.px_size = 0.032
        thread.intensity_threshold = 3.9
        thread.run()
        self.painter()

    def test_profile_collector(self):
        red = np.ones(500)*2
        green = np.ones(500)*2
        blue = np.ones(500)*2
        gen = profile_collector(self.save_path,1)
        gen.send(red)
        gen.send((red,green))
        gen.send([red,green,blue])
        try:
            gen.send(None)
        except StopIteration as exc:
            profiles = exc.value
            print(f"valid finish with {profiles}")
        self.assertTrue(os.path.exists(self.save_path+r"\red1.txt"))
        self.assertTrue(os.path.exists(self.save_path+r"\green1.txt"))
        self.assertTrue(os.path.exists(self.save_path+r"\blue1.txt"))
        os.remove(self.save_path + r'\red1.txt')
        os.remove(self.save_path + r'\green1.txt')
        os.remove(self.save_path + r'\blue1.txt')

    def test_profile_painter(self):
        line = {'X':[],'Y':[]}
        line['X'] = np.linspace(0,self.data.shape[-2]-1,600)
        line['Y'] = np.linspace(0,self.data.shape[-1]-1,600)
        data = self.data[0, 0]
        data = cv2.cvtColor(data.astype(np.uint16), cv2.COLOR_GRAY2RGBA)
        painter = profile_painter(data, self.save_path)
        painter.send(line)
        try:
            painter.send(None)
        except StopIteration:
            print("valid finish")
        self.painter()


    def tearDown(self):
        pass




if __name__ == '__main__':
    unittest.main()