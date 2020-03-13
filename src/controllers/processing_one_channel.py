from controllers.utility import *
from controllers.processing_template import QSuperThread
from controllers.micro_services import profile_painter, profile_collector

class QProcessThread(QSuperThread):
    """
    Processing thread to compute distances between SNCs in a given SIM image.
    Extending the QThread class keeps the GUI running while the evaluation runs in the background.
    """
    def __init__(self, *args, parent=None):
        super(QProcessThread, self).__init__(*args, parent)
        self.upper_limit = 800
        self.lower_limit = 400
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
        processing_image = np.clip(self.current_image[0]/self.intensity_threshold, 0, 255).astype(np.uint8)
        # spline fit skeletonized image
        self.gradient_table, self.shapes = get_center_of_mass_splines(
            processing_image, self.blur)
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
        #cv2.imshow("asadf", self.im_floodfill_inv)
        #cv2.waitKey(0)

        spline_positions = self.gradient_table[:,0:2]

        # compute new gradient table with adjusted line shape
        # indices = []
        # index = 0
        # to_cut = 0
        # running_index = 0
        # for i in range(self.gradient_table.shape[0]):
        #     running_index += 1
        #     shape = self.shapes[index]
        #     if self.im_floodfill_inv[spline_positions[i,0].astype(np.uint32), spline_positions[i,1].astype(np.uint32)]==0:
        #         indices.append(i)
        #         to_cut +=1
        #     if running_index == shape:
        #         self.shapes[index] -= to_cut
        #         to_cut=0
        #         index +=1
        #         running_index = 0
        #
        # self.gradient_table = np.delete(self.gradient_table, np.array(indices).astype(np.uint32), axis=0)

        # fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharey=True)
        # image = self.current_image[1] / self.intensity_threshold
        # axs[0].imshow(image)
        # axs[0].set_xlabel("test_image")
        # axs[1].imshow(image)
        # axs[1].set_xlabel("test_image with fitted splines")
        # #spline_table, shapes = self.gradient_table
        # spline_positions = self.gradient_table[:, 0:2]
        # axs[1].plot(spline_positions[:,1],
        #             spline_positions[:,0], c="r")
        #
        # plt.show()
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
        return profile_mean

    def _show_profiles(self):
        """
        Create and evaluate line profiles.
        """
        #line_profiles_raw = np.zeros_like(self.image_RGB)

        counter = -1
        count = self.gradient_table.shape[0]
        self.results = {'p_red': [], 'p_green': [], 'p_blue': [], 'distances': []}
        current_profile_width = int(2*self.profil_width/3*self.px_size*1000)
        if current_profile_width % 2 != 0:
            current_profile_width += 1

        painter = profile_painter(self.current_image[0]/self.intensity_threshold, self.path)
        for i in range(len(self.shapes)):
            color = self.colormap(i/len(self.shapes))
            #collector = profile_collector(self.path, i)

            current_profile = {'red':[], 'green':[]}


            for j in range(self.shapes[i]):
                counter+=1
                self.sig.emit(int((counter) / count * 100))

                # profile line starting at point walking in gradient direction
                source_point = self.gradient_table[counter,0:2]

                gradient = self.gradient_table[counter,2:4]
                gradient = np.arctan(gradient[1] / gradient[0]) + np.pi / 2

                line = line_parameters(source_point, gradient, self.profil_width)

                profile = line_profile(self.current_image[0], line['start'], line['end'], px_size=self.px_size, sampling=self.sampling)

                try:
                    distance, center = calc_peak_distance(profile)
                except:
                    print("could not fit")
                    continue

                if distance < self.lower_limit or distance> self.upper_limit:
                    continue


                profile = profile[int(center-current_profile_width/2):int(center+current_profile_width/2)]
                if profile.shape[0] < current_profile_width:
                    continue
                current_profile['red'].append(profile)
                self.results['distances'].append(distance)

                # draw line
                painter.send((line,color))

            if current_profile['red']:
                self.results['p_red'] += current_profile['red']
                red_mean = self.save_avg_profile(current_profile['red'], "red_"+str(i))
                self.sig_plot_data.emit(red_mean, current_profile_width/2, i, self.path,
                                 color, len(current_profile['red']))

        try:
            painter.send(None)
        except StopIteration:
            print("Overlay sucess")

        red_mean = self.save_avg_profile(self.results['p_red'],"red_mean")


        self.sig_plot_data.emit(red_mean, current_profile_width/2, 9999, self.path,
                                (1.0, 0.0, 0.0, 1.0), len(self.results['p_red']))
        #todo: plot histogram data

        # save profiles and distances
        np.savetxt(self.path + r"\red.txt", np.asarray(self.results['p_red']).T)
        np.savetxt(self.path + r"\distances" + ".txt", np.array(self.results['distances']))

        #cv2.imshow("Line Profiles", self.image_RGBA)

    def run(self,):
        """
        Start computation and run thread
        """
        for i in range(self.image_stack.shape[1]):
            self._z = i
            self._set_image(i)
            self._show_profiles()
        self.done.emit(self.ID)