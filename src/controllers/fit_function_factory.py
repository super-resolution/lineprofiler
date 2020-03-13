import numpy as np
from scipy import ndimage

fit_functions = dict()
def register(func):
    #print(func.__name__)
    if func.__name__ not in fit_functions:
        fit_functions[func.__name__] = func
    return func

@register
class halfnorm:
    fit_parameters = ("Intensity 1: ", "Width 1: ", "Center 1: ", "Offset: ")
    @staticmethod
    def bounds(param):
        return np.array([[0, np.inf], [0, np.inf], [0, np.inf],
                          [0, np.inf]]).T

    @staticmethod
    def guess(param):
        return [param['height'], 0.5, param['maximas'][0], 3]

    @staticmethod
    def fit(x, height, width, center, noise_lvl):
        """
        Simple guassian function.

        Parameters
        ----------
        x: ndarray
            Coordinate space in x direction
        height: float
            Maximum height of gaussian function
        center: float
            Center of gaussian funtcion
        width: float
            Width of gaussian function
        noise_lvl: float
            y offset (background lvl)

        Returns
        -------
        gaussian: ndarray
            (nx1) array of y values corresponding to the given parameters

        """
        y = np.zeros_like(x).astype(np.float32)
        indices = np.where(x-center>=0)
        y[indices] = height * np.exp(-(x[indices] - center) ** 2 / (2 * width ** 2))
        #indices = np.where(x - center < 0)
        #y[indices] = noise_lvl
        return y

@register
class gaussian:
    fit_parameters = ("Intensity 1: ", "Width 1: ", "Center 1: ", "Offset: ")
    @staticmethod
    def bounds(param):
        return np.array([[0, param['height'] + 0.1], [0, np.inf], [0, param["width"]],
                          [0, np.inf]]).T

    @staticmethod
    def guess(param):
        return [param['height']/2, 0.5, param['maximas'][0], 0]

    @staticmethod
    def fit(x, height, width, center, noise_lvl):
        """
        Simple guassian function.

        Parameters
        ----------
        x: ndarray
            Coordinate space in x direction
        height: float
            Maximum height of gaussian function
        center: float
            Center of gaussian funtcion
        width: float
            Width of gaussian function
        noise_lvl: float
            y offset (background lvl)

        Returns
        -------
        gaussian: ndarray
            (nx1) array of y values corresponding to the given parameters

        """
        return height * np.exp(-(x - center) ** 2 / (2 * width ** 2)) + noise_lvl

@register
class bigaussian:
    fit_parameters = ("Intensity 1: ", "Width 1: ", "Center 1: ",
                      "Intensity 2: ", "Width 2: ", "Center 2: ", "Offset: ")
    @staticmethod
    def bounds(param):
        return np.array([[0, param['height'] + 0.1], [0, np.inf], [0, param["width"]],
                        [0, param['height'] + 0.1], [0, np.inf], [0, param['width']],
                         [0, 0.1]]).T

    @staticmethod
    def guess(param):
        return [param['height']/2, 0.5, param['maximas'][0],
                param['height']/2, 0.5, param['maximas'][0], 0]

    @staticmethod
    def fit(x, h1, w1, c1, h2, w2, c2, noise_lvl):
        """
        Sum of two gaussian functions.

        Parameters
        ----------
        See :func: `gaussian`


        Returns
        -------
        gaussian: ndarray
            (nx1) array of y values corresponding to the given parameters
        """
        return (gaussian.fit(x, h1, w1, c1, 0) +
                gaussian.fit(x, h2, w2, c2, 0) + noise_lvl)

@register
class trigaussian:
    fit_parameters = ("Intensity 1: ", "Width 1: ", "Center 1: ",
                      "Intensity 2: ", "Width 2: ", "Center 2: ",
                      "Intensity 3: ", "Width 3: ", "Center 3: ", "Offset: ")

    @staticmethod
    def bounds(param):
        return np.array([[0, param['height'] + 0.1], [0, np.inf], [0, param["width"]],
                         [0, param['height'] + 0.1], [0, np.inf], [0, param['width']],
                         [0, param['height'] + 0.1], [0, np.inf], [0, param['width']],
                         [0, 0.1]]).T

    @staticmethod
    def guess(param):
        return [param['height']/2, 0.5, param['maximas'][0],
                param['height']/2, 0.5, param['maximas'][0],
                param['height'] / 2, 0.5, param['maximas'][0],0]

    @staticmethod
    def fit(x, h1, w1, c1, h2, w2, c2, h3, w3, c3, noise_lvl):
        """
        Sum of three gaussian function.

        Parameters
        ----------
        See :func: `gaussian`


        Returns
        -------
        gaussian: ndarray
            (nx1) array of y values corresponding to the given parameters
        """
        return (gaussian.fit(x, h1, w1, c1, 0) +
                gaussian.fit(x, h2, w2, c2, 0) +
                gaussian.fit(x, h3, w3, c3, 0) + noise_lvl)

@register
class cylinder_projection():
    fit_parameters = ("Intensity cylinder: ", "Center: ", "Inner Radius: ", "Outer Radius: ", "Offset: ", "Blur: ")

    @staticmethod
    def bounds(param):
        return np.array([[0, np.inf], [param['CM']-10, param['CM']+10],
                            [8*param['expansion'], 40*param['expansion']],
                         [15*param['expansion'], 60*param['expansion']], [0,0.01],[9,11]]).T

    @staticmethod
    def guess(param):
        r_multi = param['expansion']/2
        return [2 * param['height'], param['CM'], 25 * r_multi + 8.75, 25 * r_multi + 8.75 * 2, 0, 10]

    @staticmethod
    def fit(x, intensity, center, r1, r2, offset, blur):
        """
        Intensity projection of a cylinder

        Parameters
        ----------
        x: ndarray
            Coordinate space in x direction
        center: float
            Center coordinate of cylinder function
        intensity: float
            Maximum intensity of cylinder projection
        r1: float
            Inner radius
        r2: float
            Outer radius
        blur: flaot
            Convolve cylinder projection with gaussian blur of size "blur"

        Returns
        -------
        y: ndarray
            y values of cylinder function
        """
        axis = x
        axis = axis-int(center)
        axis = np.abs(axis)
        y = np.zeros_like(axis)
        pos1 = np.where(axis<r1)
        pos2 = np.where(np.logical_and(axis>=r1, axis<r2))
        y[pos1] = np.sqrt(r2 ** 2 - axis[pos1] ** 2) - np.sqrt(r1 ** 2 - axis[pos1] ** 2)
        y[pos2] = np.sqrt(r2 ** 2 - axis[pos2] ** 2)
        y = y/r2
        y = y*intensity
        y = y+offset
        if blur is not None:
            y = ndimage.gaussian_filter1d(y, sigma=blur)
        return y

@register
class multi_cylinder_projection:
    fit_parameters = ("Intensity cylinder 1: ", "Intensity cylinder 2: ", "Intensity cylinder 3: ",
                      "Center: ", "Expansion: ", "Offset: ", "Sigma: ")

    @staticmethod
    def bounds(param):
        return np.array([[0, np.inf], [0, np.inf], [0, np.inf], [param['CM']-10, param['CM']+10],
                         [param['expansion']-0.5, param['expansion']+0.5],[0,5], [5,15]]).T

    @staticmethod
    def guess(param):
        return [param['height']/2, param['height']/3, param['height'], param['CM'], param['expansion'], 0, 10]

    @staticmethod
    def fit(x, i1, i2, i3, c, expansion, offset, blur):
        cyl1 = cylinder_projection.fit(x, i1, c, 25*expansion/2-8.75*2, 25*expansion/2-8.75, 0, blur)
        cyl2 = cylinder_projection.fit(x, i2, c, 42.5*expansion/2, 42.5*expansion/2+8.75, 0, blur)
        cyl3 = cylinder_projection.fit(x, i3, c, 25*expansion/2+8.75,25*expansion/2+8.75*2, 0, blur)

        return (cyl1+cyl2+cyl3+offset)