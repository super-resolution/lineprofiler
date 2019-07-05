import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread,pyqtSignal
#import numba
import tifffile
import os
from controllers.utility import *
import matplotlib



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
        self.blur = 20
        self._distance_transform_th = 0.4#0.85
        self.intensity_threshold = 30



    def set_data(self, image_stack, px_size):
        self.image_stack = image_stack[0:3]
        self.results = np.zeros((self.image_stack.shape[1],2))
        self.px_size = px_size
        self.profiles = []
        self.profiles_green = []
        self.profiles_blue = []
        self.mean_distance = []
        self.distances = []
        self.images_RGB = []
        self.sampling = 10


    def _set_image(self, slice):
        """
        preprocess image

        Parameters
        ----------
        slice: int
            Current slice of image stack

        """
        self.current_image = self.image_stack[1,slice].astype(np.uint16)
        self.image = np.clip(self.image_stack[1,slice]*40/self.intensity_threshold, 0, 255).astype(np.uint8)


        self.candidates = np.zeros((self.image.shape[0],self.image.shape[1]))
        self.candidate_indices = np.zeros((1))
        self.image_RGB = cv2.cvtColor(self.current_image,cv2.COLOR_GRAY2RGB).astype(np.uint16)*200
        # spline fit skeletonized image
        self.gradient_table = compute_line_orientation(self.image, self.blur)



    def _line_profile(self, image, start, end):
        num = np.linalg.norm(np.array(start) - np.array(end)) * self.px_size*100*self.sampling
        x, y = np.linspace(start[0], end[0], num), np.linspace(start[1], end[1], num)

        return scipy.ndimage.map_coordinates(image, np.vstack((x, y)))



    def _show_profiles(self):
        """
        create and align line profiles of candidate indices
        :return: line profiles and their position in a RGB image
        """
        #line_profiles_raw = np.zeros_like(self.image_RGB)
        counter = 0
        count = self.gradient_table.shape[0]

        for j in range(self.gradient_table.shape[0]):
            points = self.gradient_table[j,0:2]
            gradients = self.gradient_table[j,2:4]
            #for i in range(points.shape[0]):
            counter+=1
            print(j)

            k, l = points[0],points[1]
            gradient = np.arctan(gradients[1]/gradients[0])+np.pi/2

            x_i = -30 * np.cos(gradient)
            y_i = -30 * np.sin(gradient)
            start = [k - x_i, l - y_i]
            end = [k +   x_i, l +  y_i]
            num = np.sqrt(x_i ** 2 + y_i ** 2)
            profile = self._line_profile(self.current_image, start, end)
            try:
                print(np.argmax(profile))
                x = np.where(profile>0)[0][0]
                profile = profile[150:650]
            except:
                continue
            if profile.shape[0]!=500:
                print("to short")
                continue

            #profile = np.delete(profile, [range(start)], 0)[0:110*self.sampling]
            self.profiles.append(profile)


            x, y = np.linspace(k - x_i, k +  x_i, 3*num), np.linspace(l - y_i, l + y_i, 3*num)
            try:
                self.image_RGB[x.astype(np.int32), y.astype(np.int32)] = np.array([50000,0, 0 ])
            except:
                print("out of bounds")
            self.sig.emit(int((counter) / count* 100))

            #line_profiles_raw[x.astype(np.int32), y.astype(np.int32)] = np.array([50000, 0, 0])
        self.images_RGB.append(self.image_RGB)
        cv2.imshow("asdf", self.image_RGB)
        #self.images_RGB.append(line_profiles_raw)

    def run(self):
        # distance transform threshold candidates
        try:
            for i in range(self.image_stack.shape[1]):
                self._z = i
                if True:
                    self._set_image(i)
                    self._show_profiles()

                else:
                    print("nothing found in layer ", i)
            tifffile.imwrite(os.getcwd()+r'\data\profiles.tif', np.asarray(self.images_RGB).astype(np.uint16), photometric='rgb')

            red = np.array(self.profiles)
            red_mean = np.mean(red, axis=0)
            np.savetxt(os.getcwd() + r"\data\red_mean.txt", red_mean)
            self.fit_data(red_mean)
            plt.plot(red_mean,"r")
            plt.show()
            np.savetxt(os.getcwd()+r"\data\red.txt",red)
        except:
            raise
        finally:
            self.done.emit()

    def fit_data(self, data):
        x = np.linspace(0, data.shape[0]-1, data.shape[0])
        optim = fit_data_to_gaussian(data)

        matplotlib.rc('font', **{'size' : 12},)
        matplotlib.rcParams['font.sans-serif'] = "Helvetica"
        x_aligned = x-30 * self.px_size * 100 * self.sampling+150
        plt.plot(x_aligned, gaussian(x, optim[0],optim[1],optim[2],optim[-1])/data.max(),
                 lw=1, c='r', ls='--', label='bi-Gaussian fit')
        plt.plot(x_aligned, gaussian(x, optim[3], optim[4], optim[5], optim[-1])/data.max(),
                 lw=1, c='r', ls='--', )
        #plt.plot(x_aligned, gaussian(x, optim[6], optim[7], optim[8], optim[-1])/data.max(),
        #         lw=1, c='r', ls='--', )
        plt.plot(x_aligned, data/data.max(), label="averaged line profile")
        plt.legend(loc='best')
        plt.ylabel("normed intensity [a.u.]")
        plt.xlabel("distance [nm]")# coordinate space perpendicular to spline fit

        print("distance = ", optim[1]-optim[4])
        print("offset = ", optim[-1])
        plt.show()

    @property
    def distance_transform_th(self):
        return self._distance_transform_th

    @distance_transform_th.setter
    def distance_transform_th(self, value):
        self._distance_transform_th = value