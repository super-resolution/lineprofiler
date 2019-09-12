import cv2
import numpy as np
import scipy
#import matplotlib.pyplot as plt
import tifffile
from controllers.utility import *
from controllers.processing import QSuperThread



class QProcessThread(QSuperThread):
    """
    Processing thread to compute distances between SNCs in a given SIM image.
    Extending the QThread class keeps the GUI running while the evaluation runs in the background.
    """
    def __init__(self, parent=None):
        super(QProcessThread, self).__init__(parent)
        self.upper_lim = 800
        self.lower_lim = 400
        self._distance_transform_th = 0.4#0.85
        self.distance_to_center = 900

    def set_image(self, slice):
        """
        Set current image
        :param slice: z-slice of the hyperstack
        """
        self.current_image_r = self.image_stack[0,slice]
        self.image = np.clip(self.image_stack[0,slice]/self._intensity_threshold, 0, 255).astype(np.uint8)
        cv2.imshow("asdf", self.image)
        #cv2.waitKey(0)
        #self.current_image_green = self.image_stack[1,slice]
        #self.image_green = np.clip(self.image_stack[1,slice]*2.5/self._intensity_threshold, 0, 255).astype(np.uint8)
        self.two_channel = False
        if self.image_stack.shape[0] == 3:
            self.current_image_blue = self.image_stack[2,slice]
        else:
            self.two_channel = True
            print("no blue channel")
        self.candidates = np.zeros((self.image.shape[0],self.image.shape[1]))
        self.candidate_indices = np.zeros((1))
        self.image_RGBA = np.zeros((self.current_image_r.shape[0], self.current_image_r.shape[1], 4)).astype(np.uint16)
        self._init_distance_transform()
        self.grad_image = create_gradient_image(self.image, self._blur)

    def _init_distance_transform(self):
        """
        Create distance transform of closed shapes in the current self.image
        """
        image = self.image
        image = cv2.blur(image, (self._blur, self._blur))

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
        cv2.imshow("asdf",cv2.resize(self.im_floodfill_inv,(0,0), fx=0.5, fy=0.5))
        cv2.imshow("asdfg", self.image_canny)
        cv2.waitKey(0)


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
        current_profile = []
        for i in range(self.candidate_indices.shape[1]):
            print(i)
            k, l = self.candidate_indices[0, i], self.candidate_indices[1, i]
            gradient = self.grad_image[k, l]

            x_i = -40 * np.cos(gradient)
            y_i = -40 * np.sin(gradient)
            start = [k - x_i, l - y_i]
            end = [k +  x_i, l +  y_i]
            num = np.sqrt(x_i ** 2 + y_i ** 2)
            profile = line_profile(self.current_image_r, start, end, px_size=self.px_size, sampling=self.sampling)
            #profile_green = self._line_profile(self.current_image_green, start, end)
            #if not self.two_channel:
            #    profile_blue = self._line_profile(self.current_image_blue, start, end)
            # try:
            #     # x = np.where(profile>0)[0][0]
            #     profile = profile[int(50 * self.px_size * 100):int(550 * self.px_size * 100)]
            # except:
            #     continue
            if profile.shape[0] < 499 * self.px_size * 100:
                print("to short")
                continue

            distance, center = calc_peak_distance(profile)
            if distance < self.lower_lim or distance > self.upper_lim:
                continue

            profile = profile[int(center - self.distance_to_center):int(center + self.distance_to_center)]
            if profile.shape[0] != 2*self.distance_to_center:
                continue

            self.profiles.append(profile)
            current_profile.append(profile)

            x, y = np.linspace(k - x_i, k + x_i, 3 * num), np.linspace(l - y_i, l + y_i, 3 * num)
            if x.min() > 0 and y.min() > 0 and x.max() < self.image_RGBA.shape[0] and y.max() < self.image_RGBA.shape[
                1]:
                self.image_RGBA[x.astype(np.int32), y.astype(np.int32)] = np.array(
                    [1.0, 0, 0, 1.0]) * 50000
            else:
                print("out of bounds")

        if current_profile:
            red = np.array(current_profile)
            red_mean = np.mean(red, axis=0)
            np.savetxt(self.path+r"\red_"+str(i)+".txt",red_mean)
            self.sig_plot_data.emit(red_mean, self.distance_to_center, i, self.path,
                                    (1.0, 0, 0, 1.0), red.shape[0])


            #line_profiles_raw[x.astype(np.int32), y.astype(np.int32)] = np.array([50000, 0, 0])
        cv2.imshow("asdf", self.image_RGBA)
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
                self.candidate_indices = np.array(np.where(dis_transform != 0))

                #self._compute_orientation()
                get_candidates_accelerated(maximum, dis_transform, self.image_canny, self.candidates, self._distance_transform_th)
                self.candidate_indices = np.array(np.where(self.candidates != 0))
                self.show_profiles()

            else:
                print("nothing found in layer ", i)
            self.sig.emit(int((i+1)/self.image_stack.shape[1]*100))
        self.done.emit()
        tifffile.imwrite(self.path + r'\profiles.tif', self.image_RGBA.astype(np.uint16), photometric='rgb')
        new = np.zeros((self.current_image_r.shape[0], self.current_image_r.shape[1], 3))
        new[..., 0] = self.current_image_r
        new[..., 1] = self.current_image_r
        new[..., 2] = self.current_image_r

        new += np.asarray(self.image_RGBA)[...,0:3]
        new = np.clip(new, 0, 65535)
        tifffile.imwrite(self.path + r'\Image_overlay.tif', new[..., 0:3].astype(np.uint16), photometric='rgb')

        distanc = np.asarray(self.distances)
        np.savetxt(self.path + "distances.txt",distanc)
        histogram = np.histogram(self.distances, bins=np.linspace(self.lower_lim,self.upper_lim,(self.upper_lim-self.lower_lim)/10+1),)
        z = int((self.upper_lim-self.lower_lim)/10)
        hist = np.zeros((z,3))
        for i in range(hist.shape[0]):
            hist[i,0] = histogram[0][i]
            hist[i,1] = histogram[1][i]
            hist[i,1] = histogram[1][i+1]

        np.savetxt(self.path + r"\distances_histogram.txt",hist.astype(np.int16))
        #plt.hist(self.distances, bins=np.linspace(self.lower_lim,self.upper_lim,(self.upper_lim-self.lower_lim)/10+1))
        #plt.savefig(self.path + 'Histogram.png')
        #plt.suptitle('Histogram', fontsize=16)
        #plt.show()
        file = open(self.path + "results.txt", "w")
        file.write("mean distance is: "+ str(np.mean(distanc))+ "\nste is: "+ str(np.std(distanc)/np.sqrt(len(distanc))))
        file.close()
        print("mean distance is:",np.mean(distanc))
        print("ste is:", np.std(distanc)/np.sqrt(len(distanc)))
        red = np.array(self.profiles)
        np.savetxt(self.path +  "red.txt",red)
        red_mean = np.mean(red, axis=0)
        self.sig_plot_data.emit(red_mean, 555, 9999, self.path,
                                (1.0, 0.0, 0.0, 1.0), red.shape[0])
        #plt.plot(red_mean,"r")
        #green = np.array(self.profiles_green)
        #np.savetxt(self.path + "green.txt",green)
        #green_mean = np.mean(green, axis=0)
        #plt.plot(green_mean, "g")
        # if not self.two_channel:
        #     blue = np.array(self.profiles_blue)
        #     np.savetxt(self.path + "blue.txt",blue)
        #     blue_mean = np.mean(blue, axis=0)
        #     all_profiles = np.swapaxes(np.array([red_mean, green_mean,blue_mean,]),0,1)
        #     all_profiles_normed = np.swapaxes(np.array([red_mean/np.linalg.norm(red_mean), green_mean/np.linalg.norm(green_mean),blue_mean/np.linalg.norm(blue_mean),]),0,1)
        #     plt.plot(blue_mean, "b")
        # else:
        #     all_profiles = np.swapaxes(np.array([red_mean, green_mean,]),0,1)
        #     all_profiles_normed = np.swapaxes(np.array([red_mean/np.linalg.norm(red_mean), green_mean/np.linalg.norm(green_mean),]),0,1)
        np.savetxt(self.path + "mean_red.txt",red_mean)
        #np.savetxt(self.path + "mean_line_profiles_normed.txt",all_profiles_normed)
        #plt.savefig(self.path + "Profiles.png")
        #plt.show()

    @property
    def distance_transform_th(self):
        return self._distance_transform_th

    @distance_transform_th.setter
    def distance_transform_th(self, value):
        self._distance_transform_th = value