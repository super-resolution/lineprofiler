import tifffile
from controllers.utility import *
from controllers.processing import QSuperThread
from controllers.profile_handler import profile_painter


class QProcessThread(QSuperThread):
    """
    Processing thread to compute distances between SNCs in a given SIM image.
    Extending the QThread class keeps the GUI running while the evaluation runs in the background.
    """
    def __init__(self, *args, parent=None):
        super(QProcessThread, self).__init__(*args, parent)
        self.upper_lim = 800
        self.lower_lim = 400
        self.distance_to_center = 900
        self.three_channel = True

    def _set_image(self, slice):
        """
        Preprocess image

        Parameters
        ----------
        slice: int
            Current slice of image stack

        """
        self.current_image = self.image_stack[:,slice].astype(np.uint16)
        # use green image for line orientation
        processing_image = np.clip(self.current_image[1]/self.intensity_threshold, 0, 255).astype(np.uint8)
        # spline fit skeletonized image
        self.gradient_table, self.shapes = compute_line_orientation(
            processing_image, self.blur, expansion=self.spline_parameter, expansion2=self.spline_parameter)
        self._fillhole_image()



    def _fillhole_image(self):
        """
        Build a fillhole image in red channel

        Returns
        -------

        """
        image = self.current_image[0]/self._intensity_threshold
        image = np.clip(image,0,255)

        self.im_floodfill_inv = create_floodfill_image(image)

        spline_positions = self.gradient_table[:,0:2]

        # compute new gradient table with adjusted line shape
        indices = []
        index = 0
        to_cut = 0
        running_index = 0
        for i in range(self.gradient_table.shape[0]):
            running_index += 1
            shape = self.shapes[index]
            if self.im_floodfill_inv[spline_positions[i,0].astype(np.uint32), spline_positions[i,1].astype(np.uint32)]==0:
                indices.append(i)
                to_cut +=1
            if running_index == shape:
                self.shapes[index] -= to_cut
                to_cut=0
                index +=1
                running_index = 0

        self.gradient_table = np.delete(self.gradient_table, np.array(indices).astype(np.uint32), axis=0)

        #spline_positions = self.gradient_table[:, 0:2]
        #index = 0
        #for j in range(len(self.shapes)):
            #plt.plot(spline_positions[index:index + self.shapes[j], 1], spline_positions[index:index + self.shapes[j], 0],
                        #color='red')
            #index += self.shapes[j]
        #plt.show()

    def save_avg_profile(self, profile, name):
        profile = np.array(profile)
        profile_mean = np.mean(profile, axis=0)
        X = np.arange(0,profile_mean.shape[0],1)
        to_save = np.array([X,profile_mean]).T
        np.savetxt(self.path + "\\"+name+".txt", to_save)

    def _show_profiles(self):
        """
        Create and evaluate line profiles.
        """
        #line_profiles_raw = np.zeros_like(self.image_RGB)

        counter = -1
        count = self.gradient_table.shape[0]
        self.results = {'p_red': [], 'p_green': [], 'p_blue': [], 'distances': []}

        painter = profile_painter(self.current_image/self.intensity_threshold, self.path)
        for i in range(len(self.shapes)):
            color = self.colormap(i/len(self.shapes))
            current_profile = {'red':[], 'green':[]}
            if self.three_channel:
                current_profile.setdefault('blue',[])

            for j in range(self.shapes[i]):
                counter+=1
                self.sig.emit(int((counter) / count * 100))

                # profile line starting at point walking in gradient direction
                source_point = self.gradient_table[counter,0:2]
                gradient = self.gradient_table[counter,2:4]
                gradient = np.arctan(gradient[1] / gradient[0]) + np.pi / 2

                line = line_parameters(source_point, gradient)

                profile = line_profile(self.current_image[0], line['start'], line['end'], px_size=self.px_size, sampling=self.sampling)

                if profile.shape[0]<499* self.px_size*100:
                    print("to short")
                    continue

                distance, center = calc_peak_distance(profile)
                if distance < self.lower_lim or distance> self.upper_lim:
                    continue

                profile = profile[int(center-self.distance_to_center):int(center+self.distance_to_center)]
                if profile.shape[0] != 2*self.distance_to_center:
                    continue
                current_profile['red'].append(profile)
                self.results['distances'].append(distance)

                profile_g = line_profile(self.current_image[1], line['start'], line['end'], px_size=self.px_size, sampling=self.sampling)
                profile_g = profile_g[int(center-self.distance_to_center):int(center+self.distance_to_center)]
                current_profile['green'].append(profile_g)

                if self.three_channel:
                    profile_b = line_profile(self.current_image[2], line['start'], line['end'], px_size=self.px_size,
                                             sampling=self.sampling)
                    profile_b = profile_b[int(center - self.distance_to_center):int(center + self.distance_to_center)]
                    current_profile['blue'].append(profile_b)

                # draw line
                painter.send(line)

            if current_profile['red']:
                red = np.array(current_profile['red'])
                red_mean = np.mean(red, axis=0)
                np.savetxt(self.path+r"\red_"+str(i)+".txt",red_mean)
                self.sig_plot_data.emit(red_mean, self.distance_to_center, i, self.path,
                                 color, red.shape[0])

                self.results['p_red'] += current_profile['red']
                self.results['p_green'] += current_profile['green']

                self.save_avg_profile(current_profile['red'], "red_"+str(i))
                self.save_avg_profile(current_profile['green'], "green_"+str(i))
                if self.three_channel:
                    self.results['p_blue'] += current_profile['blue']
                    self.save_avg_profile(current_profile['blue'], "blue_" + str(i))

        try:
            painter.send(None)
        except StopIteration:
            print("Overlay sucess")

        self.save_avg_profile(self.results['p_red'],"red_mean")
        self.save_avg_profile(self.results['p_green'], "green_mean")
        if self.three_channel:
            self.save_avg_profile(self.results['p_blue'], "blue_mean")

        red = np.array(self.results['p_red'])
        red_mean = np.mean(red, axis=0)
        self.sig_plot_data.emit(red_mean, self.distance_to_center, 9999, self.path,
                                (1.0, 0.0, 0.0, 1.0), red.shape[0])

        # save profiles and distances
        np.savetxt(self.path + r"\red.txt", red.T)
        np.savetxt(self.path + r"\distances_" + str(i) + ".txt", np.array(self.results['distances']))

        #cv2.imshow("Line Profiles", self.image_RGBA)

    def run(self,): #todo: don't plot in main thread
        """
        Start computation and run thread
        """
        try:
            for i in range(self.image_stack.shape[1]):
                self._z = i
                self._set_image(i)
                self._show_profiles()


        except EnvironmentError:
            raise
        finally:
            self.done.emit()
