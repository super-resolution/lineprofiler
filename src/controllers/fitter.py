import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import optimize
import matplotlib
import numpy as np
import os
from scipy import ndimage
from collections import abc,namedtuple
from controllers.fit_function_factory import *


class Fit:
    r"""
    Performs least square fitting, providing a row of fit_functions


    Attributes
    ----------
    fit_functions: list(str)
        Name of fit functions as a list of strings. Possible values are:
            **gaussian:** :math:`y = h e^{ \frac{-(x - c)^ 2 } {2w^2}} + b`
                ,where h is the intesity, c center, w width and b background noise
            **bigaussian:**  :math:`y = h_1 e^{ \frac{-(x - c_1)^ 2 } {2w_1^2}}+h_2 e^{ \frac{-(x - c_2)^ 2 } {2w_2^2}} + b`

            **trigaussian:** :math:`y = h_1 e^{ \frac{-(x - c_1)^ 2 } {2w_1^2}}+
            h_2 e^{ \frac{-(x - c_2)^ 2 } {2w_2^2}} + h_3 e^{ \frac{-(x - c_3)^ 2 } {2w_3^2}} + b`

            **cylinder_projection:**  :math:`y = \Biggl \lbrace
            {
            h(\sqrt{r_2 ^2 - (x-c) ^ 2} - \sqrt{r_1 ^ 2 - (x-c) ^ 2}),\text{ if }
            \|x\| < r_2
            \atop
            h(\sqrt{r_2 ^2 - (x-c) ^ 2}), \text{ if }
            \|x\| \geq r1, \|x\| < r_2
            }`
                ,where h denotes the intensity, c the center, :math:`r1` the inner cylinder radius, :math:`r2` the outer
                cylinder radius
            **multi_cylinder_projection:** :math:`y =  cyl(i_1, c, 25e_x/2-2a, 25e_x/2-a) +\\
            cyl(i_2, c, 42.5e_x/2, 42.5e_x/2+a) +\\
            cyl(i_3, c, 25e_x/2+a,25e_x/2+2a) + b`
                ,this function assumes that a micrutuboli sample was pre- and post labled under expansion microscopy
                (expansion factor :math:`e_x`) the second cyl(cylinder_projection) compensates for pre labled
                fluorophores while the first and last cyl fit, post labled fluorophores considering a free orientation
                of the second antibody (antibody width a = 8.75).


    Example
    -------
    >>> fitter = fit_gaussian()
    >>> X = np.linspace(0,199,200)

    >>> gaussian = fit_gaussian.gaussian(X, 7.0, 20, 100, 0)
    >>> gaussian = gaussian/gaussian.max()
    >>> bigaussian = fitter.bigaussian(X, 6.0, 20, 140, 6.0, 20, 60, 0)
    >>> bigaussian = bigaussian/bigaussian.max()
    >>> trigaussian = fitter.trigaussian(X, 6.0, 20, 140, 6.0, 20, 60, 2.0, 20, 100, 0)
    >>> trigaussian = trigaussian/trigaussian.max()
    >>> plt.plot(gaussian, label="gaussian")
    >>> plt.plot(bigaussian, label= "bigaussian")
    >>> plt.plot(trigaussian, label="trigaussian")
    >>> plt.legend(loc='best')
    >>> plt.show()

    >>> cylinder_proj = fit_gaussian.cylinder_projection(X, 25,100, 50, 60, 0,blur=1,)
    >>> cylinder_proj = cylinder_proj/cylinder_proj.max()
    >>> multicylinder = fitter.multi_cylinder_projection(X, 6, 6, 6, 100, 3, 0, blur= 1)
    >>> multicylinder = multicylinder/multicylinder.max()
    >>> plt.plot(cylinder_proj, label="cylinder-projection")
    >>> plt.plot(multicylinder, label="multicylinder")
    >>> plt.legend(loc='best')
    >>> plt.show()



        .. image:: fig/multi_gaussian.png
           :width: 49%
        .. image:: fig/cylinder.png
           :width: 49%



    """
    def __init__(self):
        self.expansion = 1

    @property
    def fit_function(self):
        return fit_functions

    @fit_function.setter
    def fit_function(self, value):
        if isinstance(value, abc.Iterable):
            unique_val = set(value)
            #self._fit_functions = unique_val
        else:
            raise ValueError("Fit functions must be of iterable type")
        for func in unique_val:
            #if func not in fit_functions:
            register(globals()[func])
        if len(fit_functions.keys()-unique_val) !=0:
            fit_functions.pop(*(fit_functions.keys()-unique_val))
        #fit_functions.remove(fit_functions-unique_val)

    def fit_data(self, data, center, nth_line=0, path=None, c=(1.0,0.0,0.0,1.0), n_profiles=0):
        """
        Fit given data to functions in fit_functions. Creates a folder for each given function in "path". A plot of
        input data the least square fit and the optimal parameters is saved as png.

        Parameters
        ----------
        data: ndarray
        px_size: float [micro meter]
        sampling: int
        nth_line: int
            Extend path name with number on batch processing
        path: str
            Output data is saved in path
        c: tuple
            Defines the color of the data plot
        n_profiles: int
            Number of interpolated profiles. Is written as text in the plot.
        """
        matplotlib.rc('font', **{'size' : 12},)
        matplotlib.rcParams['font.sans-serif'] = "Helvetica"
        x = np.linspace(0, data.shape[0]-1, data.shape[0])
        x_aligned = x-center


        for name,func in fit_functions.items():
            fig = plt.figure()
            ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
            ax1.plot(x_aligned, data / data.max(), c=c, label="averaged line profile")

            #fit = getattr(self, "fit_data_to_"+name)
            #func = getattr(self, name)
            optim, loss = self.fit_data_to(func, x, data)
            txt = name + "fit parameters: \n" + f"Number of profiles: {n_profiles} \n"
            for i,parameter in enumerate(func.fit_parameters):
                txt += parameter + f"{np.abs(optim[i]):.2f}" + "\n"
            ax1.plot(x_aligned, func.fit(x, *optim)/data.max() ,
                    lw=1, c="r", ls='--', label=name)
            ax1.legend(loc='best')
            ax1.set_ylabel("normed intensity [a.u.]")
            ax1.set_xlabel("distance [nm]")
            fig.text(0.5, 0.01, txt, ha='center')
            fig.set_size_inches(7, 12, forward=True)
            if path is not None:
                path_new = path+ r"\\"+name
                if not os.path.exists(path_new):
                    os.makedirs(path_new)
                plt.savefig(path_new +rf'\profile_{nth_line}.png')
            plt.close(fig)
        return loss

    def fit_data_to(self, func, x, data):
        """
        Fit data to given func using least square optimization. Compute and  print chi2.
        Return the optimal parameters found for two gaussians.

        Parameters
        ----------
        data: ndarray
            Given data (1d)

        Returns
        -------
        optim2: tuple
            Optimal parameters
        """
        param = {}
        param['maximas'] = self.find_maximas(data)
        param['height'] = data.max()
        param['CM'] = np.average(x, weights=data)
        param['expansion'] = self.expansion
        param['width'] = data.shape[0]
        bounds = func.bounds(param)
        guess = func.guess(param)

        # calculate error by squared distance to data
        errfunc = lambda p, x, y: (func.fit(x, *p) - y) ** 2


        result = optimize.least_squares(errfunc, guess[:], bounds=bounds, args=(x, data))
        optim = result.x

        chi1 = lambda p, x, y: ((func.fit(x, *p) - y) ** 2)/func.fit(x, *p)


        err = chi1(optim, x, data).sum()


        print(f"{func.__name__} chi2 {err}, cost {result.cost}")
        return optim, [err, result.cost]

    @staticmethod
    def find_maximas(data, n=3):
        """
        Return the n biggest local maximas of a given 1d array.

        Parameters
        ----------
        data: ndarray
            Input data
        n: int
            Number of local maximas to find

        Returns
        -------
        values: ndarray
            Indices of local maximas.
        """
        maximas = argrelextrema(data, np.greater, order=2)
        maxima_value = data[maximas]
        values = np.ones(n)
        maximum = 0
        for i in range(n):
            try:
                index = np.argmax(maxima_value)
                if maxima_value[index]< 0.7*maximum:
                    values[i] = values[0]
                    continue

                maximum = maxima_value[index]
                maxima_value[index] = 0

                values[i] = maximas[0][index]
            except:
                print("zero exception")
        return values

    # def fit_data_to_gaussian(self, func, x, data):
    #     """
    #     Fit one, two and three gaussians to given data per least square optimization. Compute and  print chi2.
    #     Return the optimal parameters found for two gaussians.
    #
    #     Parameters
    #     ----------
    #     data: ndarray
    #         Given data (1d)
    #
    #     Returns
    #     -------
    #     optim2: tuple
    #         Optimal parameters
    #     """
    #     fit_parameters = ("Intensity 1: ","Width 1: ", "Center 1: ", "Offset: ")
    #     maximas = self.find_maximas(data)
    #     height = data.max()/2
    #     guess = [height, 0.5, maximas[0], 0]
    #     bounds = np.array([[0,data.max()+0.1],[0,np.inf],[0,data.shape[0]],
    #                         [0,0.1]]).T
    #
    #     # calculate error by squared distance to data
    #     errfunc = lambda p, x, y: (func(x, *p) - y) ** 2
    #
    #
    #     result = optimize.least_squares(errfunc, guess[:], bounds=bounds, args=(x, data))
    #     optim = result.x
    #
    #     chi1 = lambda p, x, y: ((func(x, *p) - y) ** 2)/func(x, *p)
    #
    #
    #     err = chi1(optim, x, data).sum()
    #
    #
    #     print(f"one gaussian chi2 {err}, cost {result.cost}")
    #     return optim, [err, result.cost], fit_parameters
    #
    # def fit_data_to_bigaussian(self, func, x, data):
    #     """
    #     Fit one, two and three gaussians to given data per least square optimization. Compute and  print chi2.
    #     Return the optimal parameters found for two gaussians.
    #
    #     Parameters
    #     ----------
    #     data: ndarray
    #         Given data (1d)
    #
    #     Returns
    #     -------
    #     optim2: tuple
    #         Optimal parameters
    #     """
    #     fit_parameters = ("Intensity 1: ","Width 1: ", "Center 1: ",
    #                       "Intensity 2: ","Width 2: ", "Center 2: ", "Offset: ")
    #     maximas = self.find_maximas(data)
    #     print(maximas)
    #     height = data.max()/2
    #     guess2 = [height, 0.5, maximas[0],
    #               height, 0.5, maximas[1],0]
    #     bounds2 = np.array([[0,data.max()+0.1],[0,np.inf],[0,data.shape[0]],
    #                         [0, data.max()+0.1], [0, np.inf], [0,data.shape[0]],[0,0.1]]).T
    #
    #     # calculate error by squared distance to data
    #     errfunc2 = lambda p, x, y: (func(x, *p) - y) ** 2
    #
    #     result2 = optimize.least_squares(errfunc2, guess2[:], bounds=bounds2, args=(x, data))
    #     optim2 = result2.x
    #
    #     chi2 = lambda p, x, y: ((func(x, *p) - y) ** 2)/func(x, *p)
    #
    #     err2 = chi2(optim2, x, data).sum()
    #
    #     print(f"two gaussian chi2 {err2}, cost {result2.cost} \n")
    #     return optim2, [err2, result2.cost], fit_parameters
    #
    # def fit_data_to_trigaussian(self, func, x, data):
    #     """
    #     Fit one, two and three gaussians to given data per least square optimization. Compute and  print chi2.
    #     Return the optimal parameters found for two gaussians.
    #
    #     Parameters
    #     ----------
    #     data: ndarray
    #         Given data (1d)
    #
    #     Returns
    #     -------
    #     optim2: tuple
    #         Optimal parameters
    #     """
    #     fit_parameters = ("Intensity 1: ","Width 1: ", "Center 1: ",
    #                       "Intensity 2: ","Width 2: ", "Center 2: ",
    #                       "Intensity 3: ","Width 3: ", "Center 3: ", "Offset: ")
    #     maximas = self.find_maximas(data)
    #     height = data.max()/2
    #
    #     guess3 = [height, 0.5, maximas[0],
    #               height, 0.5, maximas[0],
    #               height, 0.5, maximas[0], 0]
    #     bounds3 = np.array([[0,data.max()+0.1],[0,np.inf],[0,data.shape[0]],
    #                         [0, data.max()+0.1], [0, np.inf], [0,data.shape[0]],
    #                         [0, data.max()+0.1], [0, np.inf], [0, data.shape[0]],[0,0.1]]).T
    #
    #     # calculate error by squared distance to data
    #     errfunc3 = lambda p, x, y: (func(x, *p) - y) ** 2
    #
    #     result3 = optimize.least_squares(errfunc3, guess3[:], bounds=bounds3, args=(x, data))
    #     optim3 = result3.x
    #
    #     chi3 = lambda p, x, y: ((func(x, *p) - y) ** 2)/func(x, *p)
    #
    #     err3 = chi3(optim3, x, data).sum()
    #
    #     print(f"three gaussian chi2 {err3}, cost {result3.cost}")
    #     return optim3, [err3, result3.cost], fit_parameters
    #
    # def fit_data_to_cylinder_projection(self, func, x, data):
    #     """
    #     Fit a cylinder intensity projection to given data per least square optimization. Compute and  print chi2.
    #     Return the optimal parameters.
    #
    #     Parameters
    #     ----------
    #     data: ndarray
    #         Given data (1d)
    #
    #     Returns
    #     -------
    #     optim: tuple
    #         Optimal parameters
    #     """
    #     fit_parameters = ("Intensity cylinder: ", "Center: ", "Inner Radius: ", "Outer Radius: ", "Offset: ", "Blur: ")
    #     #maximas = self.find_maximas(data)
    #     CM = np.average(x, weights=data)
    #     height = data.max()
    #     expansion = self.expansion
    #     r_multi = self.expansion/2
    #     guess = [2*height, CM, 25*r_multi+8.75, 25*r_multi+8.75*2, 0, 10]
    #     bounds = np.array([[0, np.inf], [CM-10, CM+10],
    #                         [8*expansion, 40*expansion], [15*expansion, 60*expansion], [0,0.01],[9,11]]).T
    #     # calculate error by squared distance to data
    #     errfunc = lambda p, x, y: (func(x, *p) - y) ** 2
    #
    #     result = optimize.least_squares(errfunc, guess[:], bounds=bounds, args=(x, data))
    #     guess2 = guess
    #     result2 = optimize.least_squares(errfunc, guess2[:], bounds=bounds, args=(x, data))
    #     if result.cost< result2.cost:
    #         optim = result.x
    #     else:
    #         print("smaller max width used")
    #         optim = result2.x
    #
    #     chi = lambda p, x, y: ((func(x, *p) - y) ** 2) / func(x, *p)
    #
    #     err = chi(optim, x, data).sum()
    #     print(optim[-1])
    #
    #     print(f"three gaussian chi2 {err}, cost {result.cost}")
    #     return optim, [err,result.cost], fit_parameters
    #
    # @staticmethod
    # def find_maximas(data, n=3):
    #     """
    #     Return the n biggest local maximas of a given 1d array.
    #
    #     Parameters
    #     ----------
    #     data: ndarray
    #         Input data
    #     n: int
    #         Number of local maximas to find
    #
    #     Returns
    #     -------
    #     values: ndarray
    #         Indices of local maximas.
    #     """
    #     maximas = argrelextrema(data, np.greater, order=2)
    #     maxima_value = data[maximas]
    #     values = np.ones(n)
    #     maximum = 0
    #     for i in range(n):
    #         try:
    #             index = np.argmax(maxima_value)
    #             if maxima_value[index]< 0.7*maximum:
    #                 values[i] = values[0]
    #                 continue
    #
    #             maximum = maxima_value[index]
    #             maxima_value[index] = 0
    #
    #             values[i] = maximas[0][index]
    #         except:
    #             print("zero exception")
    #     return values
    #
    # def fit_data_to_multi_cylinder_projection(self, func, x, data):
    #     """
    #     Fit a cylinder intensity projection to given data per least square optimization. Compute and  print chi2.
    #     Return the optimal parameters.
    #
    #     Parameters
    #     ----------
    #     data: ndarray
    #         Given data (1d)
    #
    #     Returns
    #     -------
    #     optim: tuple
    #         Optimal parameters to fit two gaussians to data
    #     """
    #     fit_parameters = ("Intensity cylinder 1: ","Intensity cylinder 2: ","Intensity cylinder 3: ","Center: ", "Expansion: ", "Offset: ", "Sigma: ")
    #     #maximas = self.find_maximas(data)
    #     CM = np.average(x, weights=data)
    #     height = data.max()
    #
    #     guess = [height/2, height/3, height, CM, self.expansion, 0, 10]
    #     bounds = np.array([[0, np.inf], [0, np.inf],
    #                         [0, np.inf], [CM-10, CM+10], [self.expansion-0.5, self.expansion+0.5],[0,5], [5,15]]).T
    #
    #     # calculate error by squared distance to data
    #     errfunc = lambda p, x, y: (func(x, *p) - y) ** 2
    #
    #     result = optimize.least_squares(errfunc, guess[:], bounds=bounds, args=(x, data))
    #     optim = result.x
    #
    #
    #     chi = lambda p, x, y: ((func(x, *p) - y) ** 2) / func(x, *p)
    #
    #     err = chi(optim, x, data).sum()
    #     print(optim[-1])
    #
    #     print(f"three gaussian chi2 {err}, cost {result.cost}")
    #     return optim, [err,result.cost], fit_parameters
#
# fit_functions = dict()
# def register(func):
#     #print(func.__name__)
#     if func.__name__ not in fit_functions:
#         fit_functions[func.__name__] = func
#     return func
#
# @register
# def gaussian(x, height, width, center, noise_lvl):
#     """
#     Simple guassian function.
#
#     Parameters
#     ----------
#     x: ndarray
#         Coordinate space in x direction
#     height: float
#         Maximum height of gaussian function
#     center: float
#         Center of gaussian funtcion
#     width: float
#         Width of gaussian function
#     noise_lvl: float
#         y offset (background lvl)
#
#     Returns
#     -------
#     gaussian: ndarray
#         (nx1) array of y values corresponding to the given parameters
#
#     """
#     return height * np.exp(-(x - center) ** 2 / (2 * width ** 2)) + noise_lvl
#
# @register
# def bigaussian(x, h1, w1, c1, h2, w2, c2, noise_lvl):
#     """
#     Sum of two gaussian functions.
#
#     Parameters
#     ----------
#     See :func: `gaussian`
#
#
#     Returns
#     -------
#     gaussian: ndarray
#         (nx1) array of y values corresponding to the given parameters
#     """
#     return (gaussian(x, h1, w1, c1, 0) +
#             gaussian(x, h2, w2, c2, 0) + noise_lvl)
#
# @register
# def trigaussian(x, h1, w1, c1, h2, w2, c2, h3, w3, c3, noise_lvl):
#     """
#     Sum of three gaussian function.
#
#     Parameters
#     ----------
#     See :func: `gaussian`
#
#
#     Returns
#     -------
#     gaussian: ndarray
#         (nx1) array of y values corresponding to the given parameters
#     """
#     return (gaussian(x, h1, w1, c1, 0) +
#             gaussian(x, h2, w2, c2, 0) +
#             gaussian(x, h3, w3, c3, 0) + noise_lvl)
#
# @register
# class cylinder_projection():
#     fit_parameters = ("Intensity cylinder: ", "Center: ", "Inner Radius: ", "Outer Radius: ", "Offset: ", "Blur: ")
#
#     @staticmethod
#     def bounds(CM ,expansion):
#         return np.array([[0, np.inf], [CM-10, CM+10],
#                             [8*expansion, 40*expansion], [15*expansion, 60*expansion], [0,0.01],[9,11]]).T
#
#     @staticmethod
#     def guess(height, CM, expansion):
#         r_multi = expansion/2
#         return [2 * height, CM, 25 * r_multi + 8.75, 25 * r_multi + 8.75 * 2, 0, 10]
#
#     @staticmethod
#     def fit(x, intensity, center, r1, r2, offset, blur):
#         """
#         Intensity projection of a cylinder
#
#         Parameters
#         ----------
#         x: ndarray
#             Coordinate space in x direction
#         center: float
#             Center coordinate of cylinder function
#         intensity: float
#             Maximum intensity of cylinder projection
#         r1: float
#             Inner radius
#         r2: float
#             Outer radius
#         blur: flaot
#             Convolve cylinder projection with gaussian blur of size "blur"
#
#         Returns
#         -------
#         y: ndarray
#             y values of cylinder function
#         """
#         axis = x
#         axis = axis-int(center)
#         axis = np.abs(axis)
#         y = np.zeros_like(axis)
#         pos1 = np.where(axis<r1)
#         pos2 = np.where(np.logical_and(axis>=r1, axis<r2))
#         y[pos1] = np.sqrt(r2 ** 2 - axis[pos1] ** 2) - np.sqrt(r1 ** 2 - axis[pos1] ** 2)
#         y[pos2] = np.sqrt(r2 ** 2 - axis[pos2] ** 2)
#         y = y/r2
#         y = y*intensity
#         y = y+offset
#         if blur is not None:
#             y = ndimage.gaussian_filter1d(y, sigma=blur)
#         return y
#
# @register
# def multi_cylinder_projection(x, i1, i2, i3, c, expansion, offset, blur):
#     cyl1 = cylinder_projection(x, i1, c, 25*expansion/2-8.75*2, 25*expansion/2-8.75, 0, blur)
#     cyl2 = cylinder_projection(x, i2, c, 42.5*expansion/2, 42.5*expansion/2+8.75, 0, blur)
#     cyl3 = cylinder_projection(x, i3, c, 25*expansion/2+8.75,25*expansion/2+8.75*2, 0, blur)
#
#     return (cyl1+cyl2+cyl3+offset)