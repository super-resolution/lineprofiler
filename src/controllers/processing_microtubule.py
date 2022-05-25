from controllers.utility import compute_spline, line_parameters, line_profile, line_profile_n
import numpy as np
from controllers.processing_template import QSuperThread
from controllers.micro_services import profile_painter_2, profile_collector, mic_project_generator


class QProcessThread(QSuperThread):
    """
    Processing thread to compute Microtubule profiles.
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
        if len(self.image_stack.shape) == 2:
            self.current_image = self.image_stack
        else:
            self.current_image = self.image_stack[0,slice].astype(np.uint16)

        processing_image = np.clip(self.current_image/self.intensity_threshold, 0, 255).astype(np.uint8)
        # Spline fit skeletonized image
        self.splines = compute_spline(
            processing_image, self.blur, expansion=self.spline_parameter, expansion2=self.spline_parameter)



    def _show_profiles(self):
        """
        Create and evaluate line profiles.
        """
        if not isinstance(self.data_z, np.ndarray):
            self.z_project_collection = False
        profiles = []
        counter = -1
        count = 0
        for spl in self.splines:
            count += spl.n_points
        current_profile_width = int(2*self.profil_width/3*self.px_size*1000)
        if current_profile_width % 2 != 0:
            current_profile_width += 1
        painter = profile_painter_2(self.current_image/self.intensity_threshold, self.path)
        for i,spline in enumerate(self.splines):
            color = self.colormap(i/len(self.splines))
            collector = profile_collector(self.path, i)
            mic_generator = mic_project_generator(self.path, i)

            line_profiles = spline.profiles(spline.n_points, self.profil_width, self.px_size, self.sampling)
            for line in line_profiles:
                counter+=1
                self.sig.emit(int((counter) / count* 100))


                if self.z_project_collection:
                    for z in range(self.data_z.shape[0]):
                        z_profile = line_profile_n(self.data_z[z], line)
                        mic_generator.send((z_profile, z))

                profile = line_profile_n(self.current_image, line)
                profile = profile[int(profile.shape[0]/2-current_profile_width/2):int(profile.shape[0]/2+current_profile_width/2)]

                if profile.shape[0]< int(current_profile_width):
                    print("to short")
                    continue
                collector.send(profile)
                painter.send((line, color))
            if self.z_project_collection:
                try:
                    mic_generator.send(None)
                except StopIteration as err:
                    print("created z-profile")
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
        try:
            np.savetxt(self.path + r"\red_mean.txt", red_mean)
            self.sig_plot_data.emit(red_mean, profiles[0].shape[0]/2, 9999,
                                    self.path,
                                    (1.0, 0.0, 0.0, 1.0), red.shape[0])
            np.savetxt(self.path + r"\red.txt", red)
        except ValueError:
            print(self.path + " couldnt be evaluated")


    def run(self,):
        """
        Start computation and run thread
        """
        for i in range(self.image_stack.shape[1]):
            self._set_image(i)
            self._show_profiles()
            self.done.emit(self.ID)
