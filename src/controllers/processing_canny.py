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
        self.candidate_indices = []
        self.two_channel = False


    def set_image(self, slice):
        """
        Set current image
        :param slice: z-slice of the hyperstack
        """
        self.current_image_r = self.image_stack[0,slice]
        processing_image = np.clip(self.image_stack[0,slice]/self._intensity_threshold, 0, 255).astype(np.uint8)
        self.grad_image = create_gradient_image(processing_image, self.blur)

        #cv2.imshow("asdf", processing_image)
        #cv2.waitKey(0)
        #self.current_image_green = self.image_stack[1,slice]
        #self.image_green = np.clip(self.image_stack[1,slice]*2.5/self._intensity_threshold, 0, 255).astype(np.uint8)

        if self.image_stack.shape[0] == 3:
            self.current_image_blue = self.image_stack[2,slice]
        else:
            self.two_channel = True
            print("no blue channel")

        self.candidates = np.zeros((self.current_image_r.shape[0],self.current_image_r.shape[1]))
        self._init_distance_transform(processing_image)

    def _init_distance_transform(self, image):
        """
        Create distance transform of closed shapes in the current self.image
        """
        image = cv2.blur(image, (self.blur, self.blur))
        # canny and gradient images
        self.image_canny = cv2.Canny(image, 150, 220)

        self.im_floodfill_inv = create_floodfill_image(image)

        cv2.imshow("Floodfill image",cv2.resize(self.im_floodfill_inv,(0,0), fx=0.5, fy=0.5))
        cv2.imshow("Canny image", self.image_canny)
        #cv2.waitKey(0)

    def show_profiles(self):
        """
        create and align line profiles of candidate indices
        :return: line profiles and their position in a RGB image
        """
        #line_profiles_raw = np.zeros_like(self.image_RGB)
        #print(self.candidate_indices.shape[1])
        current_profile = []
        for i in range(self.candidate_indices.shape[1]):

            source_point = self.candidate_indices[0, i], self.candidate_indices[1, i]
            gradient = self.grad_image[source_point]

            line = line_parameters(source_point, gradient)


            profile = line_profile(self.current_image_r, line['start'], line['end'], px_size=self.px_size, sampling=self.sampling)
            if profile.shape[0] < 499 * self.px_size * 100:
                print("to short")
                continue

            distance, center = calc_peak_distance(profile)
            if distance < self.lower_lim or distance > self.upper_lim:
                continue

            factor = profile.shape[0]/line['X'].shape[0]

            profile = profile[int(center - self.distance_to_center):int(center + self.distance_to_center)]

            line['X'] = line['X'][int((center - self.distance_to_center)/factor):int((center + self.distance_to_center)/factor)]
            line['Y'] = line['Y'][int((center - self.distance_to_center)/factor):int((center + self.distance_to_center)/factor)]

            if profile.shape[0] != 2*self.distance_to_center:
                continue

            self.profiles.append(profile)
            current_profile.append(profile)
            self.distances.append(distance)

            if line['X'].min() > 0 and line['Y'].min() > 0 and\
                    line['X'].max() < self.image_RGBA.shape[0] and line['Y'].max() < self.image_RGBA.shape[1]:

                self.image_RGBA[line['X'].astype(np.int32), line['Y'].astype(np.int32)] = np.array(
                    [1.0, 0, 0, 1.0]) * 50000
            else:
                print("out of bounds")

        if current_profile:
            red = np.array(current_profile)
            red_mean = np.mean(red, axis=0)
            np.savetxt(self.path+r"\red_"+str(i)+".txt",red_mean)
            self.sig_plot_data.emit(red_mean, self.distance_to_center, i, self.path,
                                    (1.0, 0, 0, 1.0), red.shape[0])


        cv2.imshow("asdf", self.image_RGBA)



    def run(self):
        # distance transform threshold candidates
        for i in range(self.image_stack.shape[1]):
            self._z = i
            if True:
                self.set_image(i)
                dis_transform = cv2.distanceTransform(self.im_floodfill_inv, cv2.DIST_L2, 5)

                maximum = int(dis_transform.max()) + 1
                dis_transform[np.where(dis_transform < self._distance_transform_th * dis_transform.max())] = 0

                #self._compute_orientation()
                get_candidates_accelerated(maximum, dis_transform, self.image_canny, self.candidates, self.distance_transform_th)
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
        np.savetxt(self.path + r"\distances.txt",distanc)
        histogram = np.histogram(self.distances, bins=np.linspace(self.lower_lim,self.upper_lim,(self.upper_lim-self.lower_lim)/10+1),)
        z = int((self.upper_lim-self.lower_lim)/10)
        hist = np.zeros((z,3))
        for i in range(hist.shape[0]):
            hist[i,0] = histogram[0][i]
            hist[i,1] = histogram[1][i]
            hist[i,1] = histogram[1][i+1]


        file = open(self.path + r"\results.txt", "w")
        file.write("mean distance is: "+ str(np.mean(distanc))+ "\nste is: "+ str(np.std(distanc)/np.sqrt(len(distanc))))
        file.close()
        print("mean distance is:",np.mean(distanc))
        print("ste is:", np.std(distanc)/np.sqrt(len(distanc)))
        red = np.array(self.profiles)
        np.savetxt(self.path +  r"\red.txt",red)
        red_mean = np.mean(red, axis=0)
        self.sig_plot_data.emit(red_mean, 555, 9999, self.path,
                                (1.0, 0.0, 0.0, 1.0), red.shape[0])

        np.savetxt(self.path + r"\mean_red.txt",red_mean)


    @property
    def distance_transform_th(self):
        return self._distance_transform_th

    @distance_transform_th.setter
    def distance_transform_th(self, value):
        self._distance_transform_th = value