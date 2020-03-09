import tifffile
from controllers.utility import *
from controllers.processing import QSuperThread
from controllers.profile_handler import profile_painter



class QProcessThread(QSuperThread):
    """
    Processing thread to compute distances between SNCs in a given SIM image.
    Extending the QThread class keeps the GUI running while the evaluation runs in the background.
    """
    def __init__(self,*args, parent=None):
        super(QProcessThread, self).__init__(*args,parent)
        self.upper_lim = 800
        self.lower_lim = 400
        self._distance_transform_th = 0.5#0.85
        self.candidate_indices = [np.array(0),np.array(0)]
        self.two_channel = False


    def set_image(self, slice):
        """
        Set current image
        :param slice: z-slice of the hyperstack
        """
        self.current_image_r = self.image_stack[1,slice]
        processing_image = np.clip(self.image_stack[1,slice]/self._intensity_threshold, 0, 255).astype(np.uint8)
        self.grad_image = create_gradient_image(processing_image, self.blur)
        self.current_subimages, self.top_left_shift = split_by_connected_components(processing_image)
        #cv2.imshow("asdf", processing_image)
        #cv2.waitKey(0)
        #self.current_image_green = self.image_stack[1,slice]
        #self.image_green = np.clip(self.image_stack[1,slice]*2.5/self._intensity_threshold, 0, 255).astype(np.uint8)

        if self.image_stack.shape[0] == 3:
            self.current_image_blue = self.image_stack[2,slice]
        else:
            self.two_channel = True
            print("no blue channel")


    def _init_distance_transform(self, image):
        """
        Create distance transform of closed shapes in the current self.image
        """
        image = cv2.blur(image, (self.blur, self.blur))
        # canny and gradient images
        self.image_canny = cv2.Canny(image, 150, 220)

        self.im_floodfill_inv = create_floodfill_image(image)

        #cv2.imshow("Floodfill image",cv2.resize(self.im_floodfill_inv,(0,0), fx=0.5, fy=0.5))
        #cv2.imshow("Canny image", self.image_canny)
        #cv2.waitKey(0)

    def show_profiles(self):
        """
        create and align line profiles of candidate indices
        :return: line profiles and their position in a RGB image
        """
        #line_profiles_raw = np.zeros_like(self.image_RGB)
        #print(self.candidate_indices.shape[1])
        painter = profile_painter(self.current_image_r/self.intensity_threshold, self.path)
        self.profiles = []
        self.distances = []
        current_profile = []
        for i in range(self.candidate_indices.shape[1]):
            self.sig.emit((i/self.candidate_indices.shape[1])*100)

            source_point = self.candidate_indices[0, i], self.candidate_indices[1, i]
            gradient = self.grad_image[source_point]

            line = line_parameters(source_point, gradient, self.profil_width)

            # profile asyncronous with asyncio
            profile = line_profile(self.current_image_r, line['start'], line['end'], px_size=self.px_size, sampling=self.sampling)
            if profile.shape[0] < 499 * self.px_size * 100:
                print("to short")
                continue

            distance, center = calc_peak_distance(profile)
            if distance < self.lower_lim or distance > self.upper_lim:
                continue

            factor = profile.shape[0]/line['X'].shape[0]

            profile = profile[int(center - self.profil_width/3*self.px_size*1000):int(center + self.profil_width/3*self.px_size*1000)]

            line['X'] = line['X'][int((center - self.profil_width/3*self.px_size*1000)/factor):int((center + self.profil_width/3*self.px_size*1000)/factor)]
            line['Y'] = line['Y'][int((center - self.profil_width/3*self.px_size*1000)/factor):int((center + self.profil_width/3*self.px_size*1000)/factor)]

            if profile.shape[0] != int(2*self.profil_width/3*self.px_size*1000):
                continue

            self.profiles.append(profile)
            current_profile.append(profile)
            self.distances.append(distance)

            painter.send(line)

        if current_profile:
            red = np.array(current_profile)
            red_mean = np.mean(red, axis=0)
            np.savetxt(self.path+r"\red_"+str(i)+".txt",red_mean)
            self.sig_plot_data.emit(red_mean, self.profil_width/3*self.px_size*1000, i, self.path,
                                    (1.0, 0, 0, 1.0), red.shape[0])

        try:
            painter.send(None)
        except StopIteration:
            print("Overlay sucess")

    def run(self):
        # distance transform threshold candidates
        try:
            self.set_image(0)
            for i in range(self.current_subimages.shape[0]):
                processing_image = self.current_subimages[i].astype(np.uint8)

                self.candidates = np.zeros((self.current_image_r.shape[0], self.current_image_r.shape[1]))
                self._init_distance_transform(processing_image)

                dis_transform = cv2.distanceTransform(self.im_floodfill_inv, cv2.DIST_L2, 5)

                maximum = int(dis_transform.max()) + 1
                dis_transform[np.where(dis_transform < self._distance_transform_th * dis_transform.max())] = 0

                #self._compute_orientation()
                get_candidates_accelerated(maximum, dis_transform, self.image_canny, self.candidates, self.distance_transform_th)
                cd = np.array(np.where(self.candidates != 0))
                if cd.size !=0:
                    cd[0] += self.top_left_shift[i][0]
                    cd[1] += self.top_left_shift[i][1]
                    self.candidate_indices[0] = np.append(self.candidate_indices[0], cd[0])
                    self.candidate_indices[1] = np.append(self.candidate_indices[1], cd[1])
            self.candidate_indices = np.array(self.candidate_indices)
            self.show_profiles()
                #self.sig.emit(int((i+1)/self.image_stack.shape[1]*100))


            distanc = np.asarray(self.distances)
            np.savetxt(self.path + r"\distances.txt",distanc)


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
        except EnvironmentError:
            raise
        finally:
            self.done.emit(self.ID)

    @property
    def distance_transform_th(self):
        return self._distance_transform_th

    @distance_transform_th.setter
    def distance_transform_th(self, value):
        self._distance_transform_th = value