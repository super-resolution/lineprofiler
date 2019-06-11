import cv2
import numpy as np
#from impro.processing.Image import MicroscopeImage as image
import scipy
import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread,pyqtSignal
import numba
import tifffile
import os
import time

class QProcessThread(QThread):
    sig = pyqtSignal(int)
    done = pyqtSignal()
    def __init__(self, parent=None):
        super(QProcessThread, self).__init__(parent)


    def set_data(self, image_stack, px_size):
        self.image_stack = image_stack[0:3]
        self._use_green = False
        self.results = np.zeros((self.image_stack.shape[1],2))
        self.px_size = 0.02
        self.profiles = []
        self.profiles_green = []
        self.profiles_blue = []
        self.mean_distance = []
        self.distances = []
        self.images_RGB = []
        self.sampling = 10
        self.intensity_threshold = 1


    def set_image(self, slice):
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
        self.image_RGB = cv2.cvtColor(self.current_image,cv2.COLOR_GRAY2RGB).astype(np.uint16)
        #intensity threshold to be accepted by profiler
        self.threshold = 1000
        self._distance_transform_th = 0.5#0.85
        self._init_distance_transform()
        self.gradient_image()

    def _init_distance_transform(self):
        image = self.image
        image = cv2.blur(image, (5, 5))

        # canny and gradient images
        self.image_canny = cv2.Canny(image, 100, 130)
        #if self._z == 62:
        #    cv2.imshow("Canny image", self.image_canny)
        #    cv2.waitKey(0)
        # build threshold image
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        #cv2.imshow("flood",thresh)
        #cv2.waitKey(0)
        # flood image to get interior forms
        im_floodfill = thresh.copy()
        mask = np.zeros((self.image.shape[0]+2, self.image.shape[1]+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        self.im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        cv2.imshow("asdf",self.im_floodfill_inv)
        cv2.imshow("Canny", self.image_canny)
        cv2.waitKey(0)

    def gradient_image(self):
        image = self.image
        image = cv2.blur(image, (5, 5))
        X = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 1, 0, ksize=9)
        Y = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 0, 1, ksize=9)
        self.grad_image = np.arctan2(X,Y)
        image_g = self.current_image_green
        image_g = cv2.blur(image_g, (5, 5))
        X = cv2.Sobel(image_g.astype(np.float64), cv2.CV_64F, 1, 0, ksize=9)
        Y = cv2.Sobel(image_g.astype(np.float64), cv2.CV_64F, 0, 1, ksize=9)
        self.grad_image_green = np.arctan2(X,Y)

    def _line_profile(self, image, start, end):
        print(self.px_size)
        num = np.linalg.norm(np.array(start) - np.array(end)) * self.px_size*100*self.sampling
        #print(num)
        x, y = np.linspace(start[0], end[0], num), np.linspace(start[1], end[1], num)

        return scipy.ndimage.map_coordinates(image, np.vstack((x, y)))

    @staticmethod
    #@numba.jit(nopython=True)
    def get_candidates(maximum, dis_transform, image_canny, canny_candidates):
        for i in range(dis_transform.shape[0]):
            for j in range(dis_transform.shape[1]):
                if dis_transform[i, j] != 0:
                    sub_array = dis_transform[i - maximum:i + maximum, j - maximum:j + maximum]
                    sub_array[np.where(sub_array != sub_array.max())] = 0
                    dis_transform[i - maximum:i + maximum, j - maximum:j + maximum] = sub_array

        # get edges with minimal distance from middle of holes
        # cv2.cvtColor(image_canny,cv2.COLOR_GRAY2RGB)
        for i in range(dis_transform.shape[0]):
            for j in range(dis_transform.shape[1]):
                if dis_transform[i, j] != 0:
                    sub_array = image_canny[i - maximum:i + maximum, j - maximum:j + maximum]
                    dis_sub = np.ones_like(sub_array).astype(np.float32) * 255
                    for k in range(sub_array.shape[0]):
                        for l in range(sub_array.shape[1]):
                            if sub_array[k, l] != 0:
                                dis_sub[k, l] = np.sqrt((k - maximum) ** 2 + (l - maximum) ** 2)
                    index = np.where(dis_sub == dis_sub.min())
                    canny_candidates[i - maximum + index[0], j - maximum + index[1]] = 1
        #return canny_candidates

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

    #@staticmethod
    def _calc_distance(self, profile):
        split1 = profile[:43*self.sampling]
        split2 = profile[43*self.sampling:]
        distance= (split2.argmax() + 43*self.sampling) - split1.argmax()
        #print("distance is:", distance)
        return split1.argmax()+distance/2,distance



    def show_profiles(self):
        for i in range(self.candidate_indices.shape[1]):
            if self._use_green:
                k, l = self.candidate_indices_green[0, i], self.candidate_indices_green[1, i]
                gradient = self.grad_image_green[k,l]

            else:
                k, l = self.candidate_indices[0, i], self.candidate_indices[1, i]
                gradient = self.grad_image[k,l]

                # m = np.tan(direction)
                #x_i = -20 * np.cos(gradient)
                #y_i = -20 * np.sin(gradient)
                # start2 = [k + x_i, l + y_i]
                # end = [k + 2 * x_i,l + 2 * y_i]
                # num = np.sqrt(x_i ** 2 + y_i ** 2)
                # x, y = np.linspace(start2[0], end[0], num), np.linspace(start2[1], end[1], num)
                # test_profile = self.image_canny[x.astype(np.int16),y.astype(np.int16)]
                # try:
                #     index = np.where(test_profile != 0)[0][0]
                #     n,m = (start2[0]+index*np.cos(gradient)).astype(np.int16),(start2[1]+index* np.sin(gradient)).astype(np.int16)
                #     gradient2 = self.grad_image[n,m]
                #     gradient = (gradient-gradient2)/2
                # except:
                #     print("asdf")


            x_i = -20 * np.cos(gradient)
            y_i = -20 * np.sin(gradient)
            start = [k - x_i, l - y_i]
            end = [k + 2 * x_i, l + 2 * y_i]
            num = np.sqrt(x_i ** 2 + y_i ** 2)
            profile = self._line_profile(self.current_image, start, end)
            #if profile.max() < 80:
            #    print("to low profile")
            #    continue
            profile_green = self._line_profile(self.current_image_green, start, end)
            #if profile_green.max() < 6000:
            #    print("to low profile green")
            #    continue
            if not self.two_channel:
                profile_blue = self._line_profile(self.current_image_blue, start, end)
            print(len(profile))
            start,distance = self._calc_distance(profile)
            start = (start-32*self.sampling).astype(np.int32)
            if start <0 or distance <100 or distance>300:
                continue
            self.distances.append(distance)
            profile = np.delete(profile, [range(start)], 0)[0:100*self.sampling]
            self.profiles.append(profile)
            profile_green = np.delete(profile_green, [range(start)], 0)[0:100*self.sampling]
            self.profiles_green.append(profile_green)
            if not self.two_channel:
                profile_blue = np.delete(profile_blue, [range(start)], 0)[0:100*self.sampling]
                self.profiles_blue.append(profile_blue)
            x, y = np.linspace(k - x_i, k + 2 * x_i, 3*num), np.linspace(l - y_i, l + 2 * y_i, 3*num)
            self.image_RGB[x.astype(np.int32), y.astype(np.int32)] = np.array([50000,0, 0 ])
            #print("profiling2",time.time()-t1)
            #plt.plot(profile, color='r')
            #plt.show()

            print(distance)
            #plt.plot(profile_green, color='g')
            #plt.plot(profile_blue, color='b')
            #print(k, l)
        self.images_RGB.append(self.image_RGB)
        #return np.array([np.mean(distanc),np.std(distanc)])


    def run(self):
        # distance transform threshold candidates
        for i in range(self.image_stack.shape[1]):
            self._z = i
            if True:
                t1 = time.time()
                self.set_image(i)
                t2 = time.time()
                #print("set_image: ", t2-t1)
                dis_transform = cv2.distanceTransform(self.im_floodfill_inv, cv2.DIST_L2, 5)
                t1 = time.time()
                #print("dis_transform: ", t1-t2)
                maximum = int(dis_transform.max()) + 1
                dis_transform[np.where(dis_transform < self._distance_transform_th * dis_transform.max())] = 0

                #self.get_candidates(maximum, dis_transform, self.image_canny, self.candidates)
                self.get_candidates_accelerated(maximum, dis_transform, self.image_canny, self.candidates, self._distance_transform_th)
                #self.get_candidates_accelerated_green(maximum, dis_transform, self._distance_transform_th)
                self.candidate_indices_green = np.array(np.where(dis_transform != 0))
                t2 = time.time()
                #print("candidates: ", t2-t1)
                self.candidate_indices = np.array(np.where(self.candidates != 0))
                self.show_profiles() #self.results[i] =
                t1 = time.time()
                #print("profiles: ", t1-t2)
            else:
                print("nothing found in layer ", i)
            self.sig.emit(int((i+1)/self.image_stack.shape[1]*100))
        self.done.emit()
        tifffile.imwrite(os.getcwd()+r'\data\profiles.tif', np.asarray(self.images_RGB).astype(np.uint16), photometric='rgb')
        distanc = np.asarray(self.distances)
        np.savetxt(os.getcwd()+r"\data\distances.txt",distanc)
        histogram = np.histogram(self.distances, bins=np.linspace(400,800,41),)
        hist = np.zeros((40,3))
        for i in range(40):
            hist[i,0] = histogram[0][i]
            hist[i,1] = histogram[1][i]
            hist[i,1] = histogram[1][i+1]

        np.savetxt(os.getcwd() + r"\data\distances_histogram.txt",hist.astype(np.int16))
        plt.hist(self.distances, bins=np.linspace(100,300,40))
        plt.show()
        print("mean distance is:",np.mean(distanc))
        print("std is:", np.std(distanc))
        #np.savetxt(os.getcwd()+r"\results.txt",self.results)
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
        plt.show()

    @property
    def distance_transform_th(self):
        return self._distance_transform_th

    @distance_transform_th.setter
    def distance_transform_th(self, value):
        self.distance_transform_th = value