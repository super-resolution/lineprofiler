import tifffile
from controllers.utility import compute_line_orientation, line_parameters, line_profile
import numpy as np
from controllers.processing import QSuperThread


class QProcessThread(QSuperThread):
    """
    Processing thread to compute distances between SNCs in a given SIM image.
    Extending the QThread class keeps the GUI running while the evaluation runs in the background.
    """
    def __init__(self, *args, parent=None):
        super(QProcessThread, self).__init__(*args, parent)

    def _set_image(self, slice):
        """
        Preprocess image

        Parameters
        ----------
        slice: int
            Current slice of image stack

        """
        self.current_image = self.image_stack[1,slice].astype(np.uint16)*10
        processing_image = np.clip(self.image_stack[1,slice]/self.intensity_threshold, 0, 255).astype(np.uint8)
        # spline fit skeletonized image
        self.gradient_table, self.shapes = compute_line_orientation(
            processing_image, self.blur, expansion=self.spline_parameter, expansion2=self.spline_parameter)



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
                self.sig.emit(int((counter) / count* 100))

                source_point = self.gradient_table[counter,0:2]
                gradient = self.gradient_table[counter,2:4]
                gradient = np.arctan(gradient[1]/gradient[0])+np.pi/2

                line = line_parameters(source_point, gradient)

                profile = line_profile(self.current_image, line['start'], line['end'], px_size=self._px_size, sampling=self.sampling)
                try:
                    #print(np.argmax(profile))
                    profile = profile[int(50*self._px_size*100):int(550*self._px_size*100)]
                except:
                    continue
                if profile.shape[0]<499*self._px_size*100:
                    print("to short")
                    continue

                self.profiles.append(profile)
                current_profile.append(profile)

                if line['X'].min() > 0 and line['Y'].min() > 0 and \
                        line['X'].max() < self.image_RGBA.shape[0] and line['Y'].max() < self.image_RGBA.shape[1]:

                    self.image_RGBA[line['X'].astype(np.int32), line['Y'].astype(np.int32)] = np.array(
                        [color]) * 50000
                else:
                    print("out of bounds")

            red = np.array(current_profile)
            red_mean = np.mean(red, axis=0)
            np.savetxt(self.path+r"\red_"+str(i)+".txt",red_mean)
            center = 30 * self.px_size * 100 * self.sampling-(50*self.px_size*100)
            self.sig_plot_data.emit(red_mean, center, i, self.path, color, red.shape[0])

        red = np.array(self.profiles)
        red_mean = np.mean(red, axis=0)
        np.savetxt(self.path + r"\red_mean.txt", red_mean)
        self.sig_plot_data.emit(red_mean, center, 9999,
                                self.path,
                                (1.0, 0.0, 0.0, 1.0), red.shape[0])
        np.savetxt(self.path + r"\red.txt", red)

        self.images_RGBA.append(self.image_RGBA)
        #cv2.imshow("asdf", self.image_RGBA)

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
        except EnvironmentError:
            raise
        finally:
            self.done.emit()
            #self.exit()