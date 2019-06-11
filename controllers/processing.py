import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread,pyqtSignal
import numba
import tifffile
import os


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
        self.blur = 5
        self._distance_transform_th = 0.4#0.85
        self.intensity_threshold = 30



    def set_data(self, image_stack, px_size):
        self.image_stack = image_stack[0:3]
        self._use_green = False
        self.results = np.zeros((self.image_stack.shape[1],2))
        self.px_size = px_size
        self.profiles = []
        self.profiles_green = []
        self.profiles_blue = []
        self.mean_distance = []
        self.distances = []
        self.images_RGB = []
        self.sampling = 10


    def set_image(self, slice):
        """
        Set current image
        :param slice: z-slice of the hyperstack
        """
        self.current_image = self.image_stack[0,slice]
        self.image = np.clip(self.image_stack[0,slice]/self.intensity_threshold, 0, 255).astype(np.uint8)
        self.current_image_green = self.image_stack[1,slice]
        self.two_channel = False
        if self.image_stack.shape[0] == 3:
            self.current_image_blue = self.image_stack[2,slice]
        else:
            self.two_channel = True
            print("no blue channel")
        self.candidates = np.zeros((self.image.shape[0],self.image.shape[1]))
        self.candidate_indices = np.zeros((1))
        self.image_RGB = cv2.cvtColor(self.current_image,cv2.COLOR_GRAY2RGB).astype(np.int32)
        #intensity threshold to be accepted by profiler
        self._init_distance_transform()
        self.gradient_image()

    def _init_distance_transform(self):
        """
        Create distance transform of closed shapes in the current self.image
        """
        image = self.image
        image = cv2.blur(image, (self.blur, self.blur))

        # canny and gradient images
        self.image_canny = cv2.Canny(image, 150, 220)
        # build threshold image
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        #cv2.imshow("flood",thresh)
        #cv2.waitKey(0)
        # flood image to get interior forms
        im_floodfill = thresh.copy()
        mask = np.zeros((self.image.shape[0]+2, self.image.shape[1]+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        cv2.floodFill(im_floodfill, mask, (self.image.shape[1]-1,self.image.shape[0]-1), 255)
        cv2.floodFill(im_floodfill, mask, (0,self.image.shape[0]-1), 255)
        cv2.floodFill(im_floodfill, mask, (self.image.shape[1]-1,0), 255)

        self.im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        cv2.imshow("asdf",self.im_floodfill_inv)
        cv2.imshow("asdfg", self.image_canny)
        cv2.waitKey(0)

    def gradient_image(self):
        image = self.image
        image = cv2.blur(image, (self.blur, self.blur))
        X = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 1, 0, ksize=9)
        Y = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 0, 1, ksize=9)
        self.grad_image = np.arctan2(X,Y)
        image_g = self.current_image_green
        image_g = cv2.blur(image_g, (5, 5))
        X = cv2.Sobel(image_g.astype(np.float64), cv2.CV_64F, 1, 0, ksize=9)
        Y = cv2.Sobel(image_g.astype(np.float64), cv2.CV_64F, 0, 1, ksize=9)
        self.grad_image_green = np.arctan2(X,Y)

    def _line_profile(self, image, start, end):
        num = np.linalg.norm(np.array(start) - np.array(end)) * self.px_size*100*self.sampling
        x, y = np.linspace(start[0], end[0], num), np.linspace(start[1], end[1], num)

        return scipy.ndimage.map_coordinates(image, np.vstack((x, y)))

    @staticmethod
    @numba.jit(nopython=True)
    def get_candidates_accelerated(maximum, dis_transform, image_canny, canny_candidates, threshold):
        sub_array = np.zeros((2*maximum, 2*maximum))
        for i in range(dis_transform.shape[0]):
            for j in range(dis_transform.shape[1]):
                if dis_transform[i, j] != 0:
                    if i+maximum > dis_transform.shape[0] or j+maximum> dis_transform.shape[1] or i-maximum<0 or j-maximum<0:
                        print("out of bounds 1")
                        continue
                    for k in range(2*maximum):
                        for l in range(2*maximum):
                            sub_array[k,l] = dis_transform[i - maximum + k, j - maximum+l]
                    max_value= sub_array.max()
                    for k in range(2*maximum):
                        for l in range(2*maximum):
                            if dis_transform[i - maximum + k, j - maximum+l] < threshold*max_value:
                                dis_transform[i - maximum + k, j - maximum+l] = 0

        # get edges with minimal distance from middle of holes
        # cv2.cvtColor(image_canny,cv2.COLOR_GRAY2RGB)
        for i in range(dis_transform.shape[0]):
            for j in range(dis_transform.shape[1]):
                if dis_transform[i, j] != 0:
                    if i+maximum > dis_transform.shape[0] or j+maximum> dis_transform.shape[1] or i-maximum<0 or j-maximum<0:
                        print("out of bounds 2")
                        continue
                    for k in range(2*maximum):
                        for l in range(2*maximum):
                            sub_array[k,l] = image_canny[i - maximum + k, j - maximum+l]
                    dis_sub = np.ones_like(sub_array).astype(np.float32) * 255
                    for k in range(sub_array.shape[0]):
                        for l in range(sub_array.shape[1]):
                            if sub_array[k, l] != 0:
                                dis_sub[k, l] = np.sqrt((k - maximum) ** 2 + (l - maximum) ** 2)
                    min_value = dis_sub.min()
                    for k in range(sub_array.shape[0]):
                        for l in range(sub_array.shape[1]):
                            if dis_sub[k, l] == min_value:
                                canny_candidates[i - maximum + k, j - maximum+l] = 1

    def _calc_distance(self, profile):
        """
        Calc distances between peek maximas
        :param profile:
        :return: middle point of the peaks, peak distance
        """
        split1 = profile[:90*self.sampling]
        split2 = profile[90*self.sampling:]
        distance= (split2.argmax() + 90*self.sampling) - split1.argmax()
        return split1.argmax()+distance/2,distance



    def show_profiles(self):
        """
        create and align line profiles of candidate indices
        :return: line profiles and their position in a RGB image
        """
        #line_profiles_raw = np.zeros_like(self.image_RGB)
        print(self.candidate_indices.shape[1])
        for i in range(self.candidate_indices.shape[1]):
            print(i)
            if self._use_green:
                k, l = self.candidate_indices_green[0, i], self.candidate_indices_green[1, i]
                gradient = self.grad_image_green[k,l]

            else:
                k, l = self.candidate_indices[0, i], self.candidate_indices[1, i]
                gradient = self.grad_image[k,l]

            x_i = -20 * np.cos(gradient)
            y_i = -20 * np.sin(gradient)
            start = [k - x_i, l - y_i]
            end = [k + 2 * x_i, l + 2 * y_i]
            num = np.sqrt(x_i ** 2 + y_i ** 2)
            profile = self._line_profile(self.current_image, start, end)
            profile_green = self._line_profile(self.current_image_green, start, end)
            if not self.two_channel:
                profile_blue = self._line_profile(self.current_image_blue, start, end)
            start,distance = self._calc_distance(profile)
            start = (start-55*self.sampling).astype(np.int32)
            if start <0 or distance <self.lower_lim or distance>self.upper_lim:
                continue
            self.distances.append(distance)
            profile = np.delete(profile, [range(start)], 0)[0:110*self.sampling]
            self.profiles.append(profile)
            profile_green = np.delete(profile_green, [range(start)], 0)[0:110*self.sampling]
            self.profiles_green.append(profile_green)
            if not self.two_channel:
                profile_blue = np.delete(profile_blue, [range(start)], 0)[0:110*self.sampling]
                self.profiles_blue.append(profile_blue)
            x, y = np.linspace(k - x_i, k + 2 * x_i, 3*num), np.linspace(l - y_i, l + 2 * y_i, 3*num)
            self.image_RGB[x.astype(np.int32), y.astype(np.int32)] = np.array([50000,0, 0 ])
            #line_profiles_raw[x.astype(np.int32), y.astype(np.int32)] = np.array([50000, 0, 0])
        self.images_RGB.append(self.image_RGB)
        cv2.imshow("asdf", self.image_RGB)
        #self.images_RGB.append(line_profiles_raw)



    def run(self):
        # distance transform threshold candidates
        for i in range(self.image_stack.shape[1]):
            self._z = i
            if True:
                self.set_image(i)
                dis_transform = cv2.distanceTransform(self.im_floodfill_inv, cv2.DIST_L2, 5)

                maximum = int(dis_transform.max()) + 1
                dis_transform[np.where(dis_transform < self._distance_transform_th * dis_transform.max())] = 0

                self.get_candidates_accelerated(maximum, dis_transform, self.image_canny, self.candidates, self._distance_transform_th)
                self.candidate_indices_green = np.array(np.where(dis_transform != 0))
                self.candidate_indices = np.array(np.where(self.candidates != 0))
                self.show_profiles()

            else:
                print("nothing found in layer ", i)
            self.sig.emit(int((i+1)/self.image_stack.shape[1]*100))
        self.done.emit()
        tifffile.imwrite(os.getcwd()+r'\data\profiles.tif', np.asarray(self.images_RGB).astype(np.uint16), photometric='rgb')
        distanc = np.asarray(self.distances)
        np.savetxt(os.getcwd()+r"\data\distances.txt",distanc)
        histogram = np.histogram(self.distances, bins=np.linspace(self.lower_lim,self.upper_lim,(self.upper_lim-self.lower_lim)/10+1),)
        z = int((self.upper_lim-self.lower_lim)/10)
        hist = np.zeros((z,3))
        for i in range(hist.shape[0]):
            hist[i,0] = histogram[0][i]
            hist[i,1] = histogram[1][i]
            hist[i,1] = histogram[1][i+1]

        np.savetxt(os.getcwd() + r"\data\distances_histogram.txt",hist.astype(np.int16))
        plt.hist(self.distances, bins=np.linspace(self.lower_lim,self.upper_lim,(self.upper_lim-self.lower_lim)/10+1))
        plt.savefig(r'data\Histogram.png')
        #plt.suptitle('Histogram', fontsize=16)
        plt.show()
        file = open(r"data\results.txt", "w")
        file.write("mean distance is: "+ str(np.mean(distanc))+ "\nste is: "+ str(np.std(distanc)/np.sqrt(len(distanc))))
        file.close()
        print("mean distance is:",np.mean(distanc))
        print("ste is:", np.std(distanc)/np.sqrt(len(distanc)))
        red = np.array(self.profiles)
        np.savetxt(os.getcwd()+r"\data\red.txt",red)
        red_mean = np.mean(red, axis=0)
        plt.plot(red_mean,"r")
        green = np.array(self.profiles_green)
        np.savetxt(os.getcwd()+r"\data\green.txt",green)
        green_mean = np.mean(green, axis=0)
        plt.plot(green_mean, "g")
        if not self.two_channel:
            blue = np.array(self.profiles_blue)
            np.savetxt(os.getcwd()+r"\data\blue.txt",blue)
            blue_mean = np.mean(blue, axis=0)
            all_profiles = np.swapaxes(np.array([red_mean, green_mean,blue_mean,]),0,1)
            all_profiles_normed = np.swapaxes(np.array([red_mean/np.linalg.norm(red_mean), green_mean/np.linalg.norm(green_mean),blue_mean/np.linalg.norm(blue_mean),]),0,1)
            plt.plot(blue_mean, "b")
        else:
            all_profiles = np.swapaxes(np.array([red_mean, green_mean,]),0,1)
            all_profiles_normed = np.swapaxes(np.array([red_mean/np.linalg.norm(red_mean), green_mean/np.linalg.norm(green_mean),]),0,1)
        np.savetxt(os.getcwd()+r"\data\mean_line_profiles.txt",all_profiles)
        np.savetxt(os.getcwd()+r"\data\mean_line_profiles_normed.txt",all_profiles_normed)
        plt.savefig(r'data\Profiles.png')
        plt.show()

    @property
    def distance_transform_th(self):
        return self._distance_transform_th

    @distance_transform_th.setter
    def distance_transform_th(self, value):
        self._distance_transform_th = value