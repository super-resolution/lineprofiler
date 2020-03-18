from unittest import TestCase
import unittest
from matplotlib import cm

from src.controllers.fitter import *
from src.controllers.utility import *
from src.controllers.image import ImageSIM
from src.controllers import processing_SNC, processing_microtubule, processing_canny, processing_one_channel
from src.controllers.micro_services import profile_collector, profile_painter
from tifffile import TiffFile
from src.controllers.micro_services import *


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
        thread = processing_SNC.QProcessThread()
        thread.set_data(self.data, self.path)
        thread.blur = 20
        thread.px_size = 0.032
        thread.intensity_threshold = 3.9
        thread.run()
        self.painter()


    def test_processing_canny(self):
        thread = processing_canny.QProcessThread()
        thread.set_data(self.data, self.path)
        thread.blur = 9
        thread.px_size = 0.032
        thread.intensity_threshold = 3.9
        thread.run()
        self.painter()

    def test_processing_microtuboli(self):
        thread = processing_microtubule.QProcessThread()
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

class TestThreadScheduler(unittest.TestCase):
    def setUp(self):
        path = r"C:\Users\biophys\PycharmProjects\Fabi\data\test_data\MAX_3Farben-X1_16um_Out_Channel Alignment-5-X1.tif"
        image = ImageSIM(path)
        image.parse()

    def test_add_file_to_list(self):
        """
        File apears in filelist if added
        Returns
        -------

        """
        pass

    def test_file_disapears_on_processig(self):
        """
        Build thread factory on run...
        File should be visible in it's own update bar

        Check processing options
        Returns
        -------

        """
        pass

    def test_scheduler_can_handle_multiple_threads(self):
        """
        Microservice for fitting and plotting
        Check for right save folder
        Thread Shuts down after processing
        File returns to file list
        Returns
        -------

        """
        pass

class TestArtificialHelixCreation():
    def setUp(self):
        strand_distance = 800 # In nanometer
        px_size = 32.24 # In nanometer
        resolution = 200
        start_point = (500,100)
        angle = np.pi/12
        x = np.arange(800)
        y = np.cos(x/50)*strand_distance/(2*px_size)+500
        y2 = np.cos(x/50+np.pi)*strand_distance/(2*px_size)+500
        strand1 = np.array([x,y])
        strand2 = np.array([x,y2])
        R = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        strand1 = np.dot(strand1.T, R.T).astype(np.int32)
        strand2 = np.dot(strand2.T, R.T).astype(np.int32)

        image = np.zeros((1000,1000)).astype(np.uint8)
        image[strand1[:,0]+200,strand1[:,1]] = 255
        image[strand2[:,0]+200,strand2[:,1]] = 255
        blur = resolution/(px_size)
        print(blur)
        blur_i = int(np.round(blur))
        self.image = cv2.blur(image, (blur_i, blur_i))

    def test_line_profile_evaluation(self):
        fitter = Fit()
        config = {"intensity_threshold":0,
                       "px_size":0.03224,
                       "blur":12,
                       "spline_parameter":0.32,
                       "upper_limit":1200,
                       "lower_limit":400,
                       "profil_width":80}
        thread = processing_one_channel.QProcessThread()
        thread.sig_plot_data.connect(fitter.fit_data)
        for k in config.keys():
            setattr(thread, k, config[k])
        path = os.path.dirname(os.getcwd()) + "\\data\\" +"\\helix_test"
        image_stack = np.zeros((3,1,*self.image.shape))
        image_stack[1,:] = self.image
        thread.set_data(0, image_stack, path)
        thread.run()

class TestHistogram():
    def test_plot(self):
        histogramer = Hist()
        path = r"C:\Users\biophys\PycharmProjects\Fabi\data\helix_test"
        data = np.loadtxt(path+"\distances.txt")
        histogramer.create_histogram(data, path=path)
    def test_collect_distances(self):
        histogramer = Hist()
        path = r"D:\Daten\Fabi\SNCRevisonEvaluatedData\Ultra-ExM"
        data = np.array([0])
        folders = [x[1] for x in os.walk(path)]
        for x in folders[0]:
            data = np.append(data,np.loadtxt(path+"\\"+x+"\distances.txt"))
        data = data[1:]
        histogramer.create_histogram(data, path=path)

class TestJansData():
    def __init__(self):
        self.path = r"D:\Daten\Jan"

        self.folders = [x[1] for x in os.walk(self.path)]
        z=0

    def test_run_thread(self):
        for folder in self.folders[0]:
            self.files = [x[2] for x in os.walk(self.path+"\\"+folder)]
            for file in self.files[0]:
                current_path = self.path +"\\"+folder + "\\" + file
                if os.path.exists(current_path + "_" + "evaluation" + ".txt"):
                    continue
                fitter = Fit()
                thread = processing_microtubule.QProcessThread()
                thread.sig_plot_data.connect(fitter.fit_data)
                if file.split(".")[-1] != "tif":
                    continue


                with TiffFile(current_path) as tif:
                    self.data = tif.asarray()
                new_data = np.zeros((2, self.data.shape[0], self.data.shape[1]+50, self.data.shape[2]+50))
                new_data[1,:,25:25+self.data.shape[1], 25:self.data.shape[2]+25] = self.data[:]
                self.data = new_data

                service = z_stack_microservice(current_path)
                fitter.service = service
                fitter.fit_function = ["gaussian"]
                thread.set_data(0,self.data, current_path)
                thread.blur = 4
                thread.px_size = 0.1984
                thread.profil_width = 15
                thread.spline_parameter = 1
                thread.intensity_threshold = 0
                thread.run()
                try:
                    service.send(None)
                except StopIteration:
                    print("success")

def mean_profile_for_condition():
    path = r"D:\Daten\Fabi\SNCRevisonEvaluatedData\dStorm"
    data = []
    folders = [x[1] for x in os.walk(path)]
    min_len = 999999
    for x in folders[0]:
        profile = np.loadtxt(path + "\\" + x + r"\red_mean.txt")
        data.append(profile[:,1])
        if profile.shape[0]< min_len:
            min_len =  profile.shape[0]
    new_data = []
    for x in data:
        x = x[int(x.shape[0]/2-min_len/2):int(x.shape[0]/2+min_len/2)]
        new_data.append(x)
    data = np.array(new_data)

    data = np.mean(data,axis=0)
    to_save = np.array([np.arange(data.shape[0]),data])
    np.savetxt(path+r"\average_profile.txt",to_save.T)

if __name__ == '__main__':
    # case = TestArtificialHelixCreation()
    # case.setUp()
    # case.test_line_profile_evaluation()
    #mean_profile_for_condition()
    case = TestHistogram()
    case.test_plot()
    #case = TestJansData()
    #case.test_run_thread()
