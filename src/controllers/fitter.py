import matplotlib
import numpy as np
import os
from scipy import ndimage
from collections import abc,namedtuple
from controllers.fit_function_factory import *
from controllers.utility import fit_data_to
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import weakref
import gc



class Fit:
    r"""
    Performs least square fitting, providing a couple of fit_functions


    Attributes
    ----------
    fit_functions: list(str)
        Name of fit functions as a list of strings. Possible values are:
            **gaussian:** :math:`y = h e^{ \frac{-(x - c)^ 2 } {2w^2}} + b`,

            where h is the intensity, c the centre, b the offset, and w the variance of the distribution.
            Optimal for single profiles.

            **bigaussian:**  :math:`y = h_1 e^{ \frac{-(x - c_1)^ 2 } {2w_1^2}}+h_2 e^{ \frac{-(x - c_2)^ 2 } {2w_2^2}} + b`.

            Optimal for profiles with dip.

            **trigaussian:** :math:`y = h_1 e^{ \frac{-(x - c_1)^ 2 } {2w_1^2}}+
            h_2 e^{ \frac{-(x - c_2)^ 2 } {2w_2^2}} + h_3 e^{ \frac{-(x - c_3)^ 2 } {2w_3^2}} + b`.

            Optimal for profiles with dip and background.

            **cylinder_projection:**  :math:`y = \Biggl \lbrace
            {
            h(\sqrt{r_2 ^2 - (x-c) ^ 2} - \sqrt{r_1 ^ 2 - (x-c) ^ 2}),\text{ if }
            \|x\| < r_2
            \atop
            h(\sqrt{r_2 ^2 - (x-c) ^ 2}), \text{ if }
            \|x\| \geq r1, \|x\| < r_2
            }`,

            where r_1, r_2 denote the inner and outer cylinder radius. Describes the theoretical intensity profile
            for microtubule. The quality of the fit strongly depends on the initial estimation of the parameters,
            due to the nonlinearity of the cylinder function.

            **multi_cylinder_projection:** :math:`y =  cyl(i_1, c, 25e_x/2-2a, 25e_x/2-a) +\\
            cyl(i_2, c, 42.5e_x/2, 42.5e_x/2+a) +\\
            cyl(i_3, c, 25e_x/2+a,25e_x/2+2a) + b`,

            this function assumes that a micrutubule sample was pre- and post labled under expansion microscopy
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
        self._service = None

    @property
    def service(self):
        return self._service

    @service.setter
    def service(self, func):
        self._service = func

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
            functions = (fit_functions.keys()-unique_val)
            for func in functions:
                fit_functions.pop(func)
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
        matplotlib.rc('font', **{'size' : 22},)
        matplotlib.rcParams['font.sans-serif'] = "Helvetica"
        x = np.linspace(0, data.shape[0]-1, data.shape[0])
        x_aligned = x-int(center)

        for name,func in fit_functions.items():
            fig = plt.figure()
            ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
            ax1.plot(x_aligned, data / data.max(), c=c, label="averaged line profile")

            #fit = getattr(self, "fit_data_to_"+name)
            #func = getattr(self, name)
            optim, loss = fit_data_to(func, x, data, expansion=self.expansion, chi_squared=True)
            if self.service:
                self.service.send((optim[1],optim[0]))
            txt = name + "fit parameters: \n" + f"Number of profiles: {n_profiles} \n"
            for i,parameter in enumerate(func.fit_parameters):
                txt += parameter + f"{np.abs(optim[i]):.2f}" + "\n"
            ax1.plot(x_aligned, func.fit(x, *optim)/data.max(),
                    lw=1, c=c, ls='--', label=name)
            ax1.legend(loc='best')
            ax1.set_ylabel("normed intensity [a.u.]")
            ax1.set_xlabel("distance [nm]")
            fig.text(0.5, 0.01, txt, ha='center')
            fig.set_size_inches(8, 14, forward=True)
            if path is not None:
                path_new = path+ r"\\"+name
                if not os.path.exists(path_new):
                    os.makedirs(path_new)
                plt.savefig(path_new +rf'\profile_{nth_line}.svg', format="svg")
            plt.close(fig)
        plt.close("all")
        x= weakref.ref(fig)
        del fig
        del ax1
        gc.collect(2)
        return loss

class Hist():
    """
    Plot data into a histogram and fit a right sided halfnorm to the histogram.
    """
    def __init__(self, image_list=None):
        if image_list:
            self.distances = np.array([0])
            for image in image_list:
                path = os.path.dirname(os.getcwd()) + "\\data\\" + os.path.splitext(os.path.basename(image.file_path))[0] + "\distances.txt"
                if os.path.exists(path):
                    self.distances = np.append(self.distances, np.loadtxt(path))
            try:
                self.distances = self.distances[1:]
            except IndexError("No distances computed yet"):
                self.distances = None

    def create_histogram(self, values=None, path=None, start=400,stop=1100):
        """

        Parameters
        ----------
        values: data to be fitted
        start: start for the plot
        stop: for the plot +100

        Returns
        -------

        """
        if values is not None and self.distances is not None:
            values = np.append(values, self.distances)
        elif values is None and self.distances is not None:
            values = self.distances
        elif values is None and self.distances is None:
            raise ValueError("No data")
        matplotlib.rc('font', **{'size' : 12},)
        matplotlib.rcParams['font.sans-serif'] = "Helvetica"
        fig = plt.figure()
        ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7),)
        bins = np.arange(start, stop, 10)
        x = ax1.hist(values, bins, color="white",  edgecolor='#626F78')
        for item in x[2]:
            item.set_height(item.get_height() / x[0].max())
        max_value = np.max(x[0])
        rect = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        threshold_slider = Slider(rect, 'fit to rightmost peak of ..% maximum', 0.0, 1.0, valinit=0.9, valstep=0.1)
        fit, = ax1.plot(1,c="r", linestyle='dashed')
        text = fig.text(0.5, 0.01, "", ha='center')
        def update(thresh):
            j=0
            for i in range(x[0].shape[0]):
               if x[0][i]>thresh*max_value:
                   j = i
            func = fit_functions["halfnorm"]
            x_bin = np.arange(x[1].shape[0]-1)
            center_guess = x_bin[j]
            optim = fit_data_to(func, x_bin, x[0], center=center_guess)
            optim[1:4] *= 10
            print(optim)
            optim[2] += start+5
            x_bin = np.arange(stop)
            values = func.fit(x_bin, *optim)
            fit.set_xdata(x_bin[np.where(values>0)])
            fit.set_ydata(values[np.where(values>0)]/x[0].max())
            text.set_text("strand distance: " + str(np.around(optim[2], 2)) + r"$\pm$" + str(np.around(optim[1], 2)))

        update(0.9)
        threshold_slider.on_changed(update)
        ax1.set_xlim(start,stop-100)
        ax1.set_ylim(0,1.1)

        ax1.set_ylabel("normed frequency [a.u.]")
        ax1.set_xlabel("distance [nm]")
        plt.show()

        if path:
            plt.savefig(path + r'\histogram.png', dpi=1200)
        #     plt.close(fig)
        # else:


