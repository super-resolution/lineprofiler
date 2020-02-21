from controllers.utility import compute_line_orientation, line_parameters, line_profile
import numpy as np
from controllers.processing import QSuperThread
from controllers.profile_handler import profile_painter, profile_collector, mic_project_generator
import tifffile


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
        if not isinstance(self.data_z, np.ndarray):
            self.z_project_collection = False
        profiles = []
        result = []
        counter = -1
        count = self.gradient_table.shape[0]
        painter = profile_painter(self.current_image/self.intensity_threshold, self.path)
        for i in range(len(self.shapes)):
            color = self.colormap(i/len(self.shapes))
            #current_profile= []
            collector = profile_collector(self.path, i)
            #todo: add if
            mic_generator = mic_project_generator(self.path, i)
            for j in range(self.shapes[i]):
                counter+=1
                self.sig.emit(int((counter) / count* 100))


                source_point = self.gradient_table[counter,0:2]
                gradient = self.gradient_table[counter,2:4]
                gradient = np.arctan(gradient[1]/gradient[0])+np.pi/2

                line = line_parameters(source_point, gradient)

                if self.z_project_collection:
                    for z in range(self.data_z.shape[0]):
                        z_profile = line_profile(self.data_z[z], line['start'], line['end'], px_size=self.px_size,
                                               sampling=1)#todo adjust sampling to one sample per pixel
                        mic_generator.send((z_profile, z))

                profile = line_profile(self.current_image, line['start'], line['end'], px_size=self.px_size, sampling=self.sampling)
                profile = profile[int(profile.shape[0]/2-250*self.px_size*100):int(profile.shape[0]/2+250*self.px_size*100)]

                if profile.shape[0]<499*self.px_size*100:
                    print("to short")
                    continue

                collector.send(profile)
                painter.send((line, color))
            #todo: add if
            try:
                mic_generator.send(None)
            except StopIteration as err:
                mic_project = err.value*100
                tifffile.imwrite(self.path + r"\mic_project"+str(i)+ ".tif", mic_project.astype(np.uint16))
                #cv2.waitKey(0)
            try:
                collector.send(None)
            except StopIteration as err:
                result = err.value
            profiles += result["red"]
            red = np.array(result["red"])
            red_mean = np.mean(red, axis=0)
            self.sig_plot_data.emit(red_mean, profiles[0].shape[0]/2, i, self.path, color, red.shape[0])

        try:
            painter.send(None)
        except StopIteration:
            print("Overlay sucess")

        red = np.array(profiles)
        red_mean = np.mean(red, axis=0)
        np.savetxt(self.path + r"\red_mean.txt", red_mean)
        self.sig_plot_data.emit(red_mean, profiles[0].shape[0]/2, 9999,
                                self.path,
                                (1.0, 0.0, 0.0, 1.0), red.shape[0])
        np.savetxt(self.path + r"\red.txt", red)

        #cv2.imshow("asdf", self.image_RGBA)

    def run(self,): #todo: don't plot in main thread
        """
        Start computation and run thread
        """
        #try:
        for i in range(self.image_stack.shape[1]):
            self._set_image(i)
            self._show_profiles()
        self.done.emit(self.ID)
        # except EnvironmentError:
        #     raise
        # finally:
        #     self.done.emit()
        #     #self.exit()