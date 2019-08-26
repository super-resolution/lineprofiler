import cv2
import numpy as np

#import numba
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
        self.distance_to_center = 900

    def _set_image(self, slice):
        """
        Preprocess image

        Parameters
        ----------
        slice: int
            Current slice of image stack

        """
        self.current_image_r = self.image_stack[0,slice].astype(np.uint16)
        self.current_image = self.image_stack[1,slice].astype(np.uint16)*10
        self.image = np.clip(self.image_stack[1,slice]/self.intensity_threshold, 0, 255).astype(np.uint8)

        #self.candidates = np.zeros((self.image.shape[0],self.image.shape[1]))
        #self.candidate_indices = np.zeros((1))
        self.image_RGBA = np.zeros((self.current_image.shape[0],self.current_image.shape[1],4)).astype(np.uint16)#cv2.cvtColor(self.current_image,cv2.COLOR_GRAY2RGBA).astype(np.uint16)*200
        # spline fit skeletonized image
        self.gradient_table,self.shapes = compute_line_orientation(self.image, self._blur, expansion=self._spline_parameter, expansion2=self._spline_parameter)
        self._fillhole_image()

    def _fillhole_image(self):
        image = self.current_image_r/self._intensity_threshold
        image = np.clip(image,0,255).astype(np.uint8)
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        #cv2.imshow("flood",thresh)
        #cv2.waitKey(0)
        # flood image to get interior forms
        im_floodfill = thresh.copy()
        mask = np.zeros((self.image.shape[0]+2, self.image.shape[1]+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        self.im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        #plt.imshow(self.im_floodfill_inv*255)
        shapes = self.shapes
        index = 0
        spline_positions = self.gradient_table[:,0:2]
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

        spline_positions = self.gradient_table[:, 0:2]
        index = 0
        for j in range(len(self.shapes)):
            #plt.plot(spline_positions[index:index + self.shapes[j], 1], spline_positions[index:index + self.shapes[j], 0],
                        #color='red')
            index += self.shapes[j]
        #plt.show()


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

                points = self.gradient_table[counter,0:2]
                #points = self.gradient_table[counter, 4:6]
                gradients = self.gradient_table[counter,2:4]
                #for i in range(points.shape[0]):
                #print(counter)

                k, l = points[0],points[1]
                gradient = np.arctan(gradients[1]/gradients[0])+np.pi/2

                x_i = -40 * np.cos(gradient)
                y_i = -40 * np.sin(gradient)
                start = [k - x_i, l - y_i]
                end = [k + x_i, l + y_i]
                num = np.sqrt(x_i ** 2 + y_i ** 2)
                profile = line_profile(self.current_image_r, start, end, px_size=self.px_size, sampling=self.sampling)
                try:
                    print(np.argmax(profile))
                    #x = np.where(profile>0)[0][0]
                    #profile = profile[int(50*self.px_size*100):int(550*self.px_size*100)]
                except:
                    continue
                if profile.shape[0]<499*self.px_size*100:
                    print("to short")
                    continue

                distance, center = calc_peak_distance(profile)
                if distance < self.lower_lim or distance> self.upper_lim:
                    continue

                profile = profile[int(center-self.distance_to_center):int(center+self.distance_to_center)]
                if profile.shape[0] != 2*self.distance_to_center:
                    continue

                self.profiles.append(profile)
                current_profile.append(profile)

                x, y = np.linspace(k - x_i, k +  x_i, 3*num), np.linspace(l - y_i, l + y_i, 3*num)
                if x.min()>0 and y.min()>0 and x.max()<self.image_RGBA.shape[0] and y.max()< self.image_RGBA.shape[1]:
                    self.image_RGBA[x.astype(np.int32), y.astype(np.int32)] = np.array([color[0],color[1], color[2], color[3]])*50000
                else:
                    print("out of bounds")
                self.sig.emit(int((counter) / count* 100))

            red = np.array(current_profile)
            red_mean = np.mean(red, axis=0)
            np.savetxt(self.path+r"\red_"+str(i)+".txt",red_mean)
            self.sig_plot_data.emit(red_mean, self.distance_to_center, i, self.path,
                             color, red.shape[0])


            #line_profiles_raw[x.astype(np.int32), y.astype(np.int32)] = np.array([50000, 0, 0])
        self.images_RGBA.append(self.image_RGBA)
        cv2.imshow("asdf", self.image_RGBA)
        #self.images_RGB.append(line_profiles_raw)

    def run(self,): #todo: don't plot in main thread
        """
        Start computation and run thread
        """
        try:
            for i in range(self.image_stack.shape[1]):
                self._z = i
                self._set_image(i)
                self._show_profiles()


            tifffile.imwrite(self.path +r'\Image_with_RGBA_profiles.tif', np.asarray(self.images_RGBA)[...,0:3].astype(np.uint16), photometric='rgb')
            new = np.zeros((self.current_image.shape[0],self.current_image.shape[1],3))
            new[...,0] = self.current_image_r
            new[...,1] = self.current_image_r
            new[...,2] = self.current_image_r

            new += np.asarray(self.images_RGBA)[0,:,:,0:3]
            new = np.clip(new, 0,65535)
            tifffile.imwrite(self.path+r'\Image_overlay.tif', new[...,0:3].astype(np.uint16), photometric='rgb')


            red = np.array(self.profiles)
            red_mean = np.mean(red, axis=0)
            np.savetxt(self.path + r"\red_mean.txt", red_mean)
            self.sig_plot_data.emit(red_mean, self.distance_to_center, 9999, self.path,
                                    (1.0, 0.0, 0.0, 1.0), red.shape[0])

            np.savetxt(self.path+r"\red.txt",red)
        except:
            raise
        finally:
            self.done.emit()
            self.exit()