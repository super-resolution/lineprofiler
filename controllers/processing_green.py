import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread,pyqtSignal
#import numba
import tifffile
import os
from controllers.utility import *
from matplotlib import cm



class QProcessThread(QThread):
    """
    Processing thread to compute distances between SNCs in a given SIM image.
    Extending the QThread class keeps the GUI running while the evaluation runs in the background.
    """
    sig = pyqtSignal(int)
    done = pyqtSignal()
    def __init__(self, parent=None):
        super(QProcessThread, self).__init__(parent)
        self.upper_lim = 800
        self.lower_lim = 400
        self.blur = 40
        self._distance_transform_th = 0.4#0.85
        self.intensity_threshold = 80
        self.fit = fit_gaussian()
        self.colormap = cm.gist_ncar



    def set_data(self, image_stack, px_size,  f_name):
        """
        Set data for processing

        Parameters
        ----------
        image_stack: ndarray
            Ndimage of shape (C,Z,X,Y)
        px_size: float
            Pixel size [micro meter]
        f_name: str
            File name to save processed data to.

        """
        self.image_stack = image_stack[0:3]
        self.results = np.zeros((self.image_stack.shape[1],2))
        self.px_size = px_size
        self.profiles = []
        self.profiles_green = []
        self.profiles_blue = []
        self.mean_distance = []
        self.distances = []
        self.images_RGBA = []
        self.sampling = 10
        path = os.getcwd() + r"\data\\" + os.path.splitext(os.path.basename(f_name))[0]
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path


    def _set_image(self, slice):
        """
        Preprocess image

        Parameters
        ----------
        slice: int
            Current slice of image stack

        """
        self.current_image = self.image_stack[1,slice].astype(np.uint16)*10
        self.image = np.clip(self.image_stack[1,slice]*200/self.intensity_threshold, 0, 255).astype(np.uint8)

        self.candidates = np.zeros((self.image.shape[0],self.image.shape[1]))
        self.candidate_indices = np.zeros((1))
        self.image_RGBA = np.zeros((self.current_image.shape[0],self.current_image.shape[1],4)).astype(np.uint16)#cv2.cvtColor(self.current_image,cv2.COLOR_GRAY2RGBA).astype(np.uint16)*200
        # spline fit skeletonized image
        self.gradient_table,self.shapes = compute_line_orientation(self.image, self.blur)



    def _show_profiles(self):
        """
        Create and evaluate line profiles.
        """
        #line_profiles_raw = np.zeros_like(self.image_RGB)
        counter = -1
        count = self.gradient_table.shape[0]
        for i in range(len(self.shapes)):
            color = self.colormap(i/len(self.shapes))
            current_profile= []
            for j in range(self.shapes[i]):
                counter+=1

                points = self.gradient_table[counter,0:2]
                gradients = self.gradient_table[counter,2:4]
                #for i in range(points.shape[0]):
                print(counter)

                k, l = points[0],points[1]
                gradient = np.arctan(gradients[1]/gradients[0])+np.pi/2

                x_i = -30 * np.cos(gradient)
                y_i = -30 * np.sin(gradient)
                start = [k - x_i, l - y_i]
                end = [k +   x_i, l +  y_i]
                num = np.sqrt(x_i ** 2 + y_i ** 2)
                profile = line_profile(self.current_image, start, end, px_size=self.px_size, sampling=self.sampling)
                try:
                    print(np.argmax(profile))
                    #x = np.where(profile>0)[0][0]
                    profile = profile[int(50*self.px_size*100):int(550*self.px_size*100)]
                except:
                    continue
                if profile.shape[0]<499*self.px_size*100:
                    print("to short")
                    continue

                #profile = np.delete(profile, [range(start)], 0)[0:110*self.sampling]
                self.profiles.append(profile)
                current_profile.append(profile)


                x, y = np.linspace(k - x_i, k +  x_i, 3*num), np.linspace(l - y_i, l + y_i, 3*num)
                if x.min()>0 and y.min()>0 and x.max()<self.image_RGBA.shape[0] and y.max()< self.image_RGBA.shape[1]:
                    self.image_RGBA[x.astype(np.int32), y.astype(np.int32)] = np.array([color[0],color[1], color[2],color[3]])*50000
                else:
                    print("out of bounds")
                self.sig.emit(int((counter) / count* 100))

            red = np.array(current_profile)
            red_mean = np.mean(red, axis=0)
            self.fit.fit_data(red_mean, self.px_size, self.sampling, i, self.path, c=color, n_profiles=red.shape[0])
            #line_profiles_raw[x.astype(np.int32), y.astype(np.int32)] = np.array([50000, 0, 0])
        self.images_RGBA.append(self.image_RGBA)
        cv2.imshow("asdf", self.image_RGBA)
        #self.images_RGB.append(line_profiles_raw)

    def run(self,): #todo: don't plot in main thread
        """
        Start computation and run thread
        """
        try:
            for i in range(self.image_stack.shape[1]):
                self._z = i
                if True:
                    self._set_image(i)
                    self._show_profiles()

                else:
                    print("nothing found in layer ", i)

            tifffile.imwrite(self.path +r'\Image_with_RGBA_profiles.tif', np.asarray(self.images_RGBA)[...,0:3].astype(np.uint16), photometric='rgb')
            new = np.zeros((self.current_image.shape[0],self.current_image.shape[1],3))
            new[...,0] = self.current_image
            new[...,1] = self.current_image
            new[...,2] = self.current_image
            new *= 300
            new += np.asarray(self.images_RGBA)[0,:,:,0:3]
            new = np.clip(new, 0,65535)
            tifffile.imwrite(self.path+r'\Image_overlay.tif', new[...,0:3].astype(np.uint16), photometric='rgb')


            red = np.array(self.profiles)
            red_mean = np.mean(red, axis=0)
            np.savetxt(self.path + r"\red_mean.txt", red_mean)
            self.fit.fit_data(red_mean, self.px_size, self.sampling, 9999, self.path)
            plt.plot(red_mean,"r")
            plt.show()
            np.savetxt(self.path+r"\red.txt",red)
        except:
            raise
        finally:
            self.done.emit()



    @property
    def distance_transform_th(self):
        return self._distance_transform_th

    @distance_transform_th.setter
    def distance_transform_th(self, value):
        self._distance_transform_th = value