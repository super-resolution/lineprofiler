"""
Utility class for image processing futering funtcions like:
 * Create gradient image
 *
 *
"""
import numba
import cv2
import numpy as np
from skimage.morphology import skeletonize_3d
from skimage.measure import label
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import UnivariateSpline
import networkx as nx
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import optimize
from scipy import ndimage
import matplotlib

def create_gradient_image(image, blur, sobel=9):
    """
    Compute the Orientation of each pixel of a given image in rad

    Parameters
    ----------
    image: ndarray
        Image data as numpy array in grayscale
    blur: int
        Blur image with a filter of blur kernel size
    sobel(optional): int
        Kernel size of the applied sobel operators

    Returns
    -------
    gradient_image: ndarray
        ndarray of the pixel orientation in a box of "sobel" size (unit = rad)

    Example
    -------
    >>> image = cv2.imread("path_to_file.png")
    >>> gradient_image = create_gradient_image(image, 3)
    """

    image_b = cv2.blur(image, (blur, blur))
    X = cv2.Sobel(image_b.astype(np.float64), cv2.CV_64F, 1, 0, ksize=sobel)
    Y = cv2.Sobel(image_b.astype(np.float64), cv2.CV_64F, 0, 1, ksize=sobel)
    return np.arctan2(X, Y)


def compute_line_orientation(image, blur):
    """
    Compute the orientation and position line resembling patterns in an image.

    The image is convolved with a gaussian blur compensating for noise discontinuity or holes.
    A thresholding algorithm (1) converts the image from grayscale to binary. Using Lees algorithm (2)
    the expanded lines are reduced to one pixel width. The pixel coordinates from all still connected lines
    are retrieved and tested for continuity. Points of discontinuity are used as breakpoints and all following
    coordinates connected to a new line. An univariate spline of degree 3 is fitted to each line. The rounded
    coordinates and their derivatives are returned in a table, together with the length of each line.

    Parameters
    ----------
    image: ndarray
        Image containing line resembling patterns
    blur: int
        Amount of blur to apply to the image. Should be in the order of magnitude of the line width in pixel.

    Returns
    -------
    gradient_fitted_table: ndarray
        X, Y position of the splines. X, Y values of the spline gradient.
    shapes: ndarray
        Lengths of the lines written in gradient_fitted_table.

    References
    ----------
    (1)  Nobuyuki Otsu: A threshold selection method from grey level histograms.
    In: IEEE Transactions on Systems, Man, and Cybernetics. New York, 9.1979, S. 62–66. ISSN 1083-4419
    (2)  T.-C. Lee, R.L. Kashyap and C.-N. Chu,
    Building skeleton models via 3-D medial surface/axis thinning algorithms.
    Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.

    Example
    -------
    .. figure::

    >>> compute_line_orientation(image, 20)

    .. figure::

    """
    image = cv2.blur(image, (blur, blur))

    # build threshold image
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)

    skeleton = skeletonize_3d((thresh / 255).astype(np.uint8)).astype(np.uint8)
    # contour = self.collapse_contours(contours)
    cv2.imshow("asdf", skeleton * 255)
    cv2.waitKey(0)

    colormap = label(skeleton, connectivity=2)
    lines = []
    for i in range(colormap.max()):
        j = i + 1
        line = np.where(colormap == j)
        if len(line[0]) > 150:
            lines.append(line)
        else:
            for k in range(line[1].shape[0]):
                skeleton[line[0][k], line[1][k]] = 0
    point_list = []
    point_fitted_list = []
    gradient_list = []
    line_itterator = -1
    line_length = len(lines)-1
    while line_itterator < line_length:
        line_itterator += 1
        print(line_itterator)
        points = np.array(lines[line_itterator]).T
        if points.shape[0] < 70:
            continue
        order = order_points_to_line(points)
        if len(order)<points.shape[0]:
            lines.append(points[len(order)+1:].T)
            line_length += 1
        points = points[order]
        #distance from one points to the next
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))

        direction_change = 9999999
        for i in range(distance.shape[0]):
            if i + 10 < distance.shape[0]:
                vec1 = points[i + 5] - points[i]
                vec2 = points[i + 10] - points[i + 5]
                direction = np.dot(vec1, vec2)
                if direction < 50:
                    direction_change = i + 5
            if distance[i] - distance[i - 1] > 10 or i > direction_change:
                distance = distance[:i]
                lines.append(points[i + 2:].T)
                points = points[:i + 1]
                line_length += 1
                break
        if points.shape[0] < 10:
            continue
        distance = np.insert(distance, 0, 0) / distance[-1]
        point_list.append(points)

        # Build a list of the spline function, one for each dimension:

        splines = [UnivariateSpline(distance, coords, k=3, s=points.shape[0] * 10) for coords in points.T]
        dsplines = [spline.derivative() for spline in splines]
        # Computed the spline for the asked distances:
        alpha = np.linspace(0, 1, points.shape[0])
        points_fitted = np.vstack(spl(alpha) for spl in splines).T
        #append fitted points to
        point_fitted_list.append(points_fitted)
        gradient_list.append(np.vstack(spl(alpha) for spl in dsplines).T)

        #plot results for testing purposes
        plt.plot(points[..., 0], points[..., 1], color="g")
        plt.plot(points_fitted[..., 0], points_fitted[..., 1])

    #sort results to array
    result_table = []
    shapes = []
    for i in range(len(point_fitted_list)):
        shapes.append(point_fitted_list[i].shape[0])
        for j in range(point_fitted_list[i].shape[0]):
            result_table.append([int(point_fitted_list[i][j][0]), int(point_fitted_list[i][j][1]),
                                 gradient_list[i][j][0], gradient_list[i][j][1],i])
    gradient_fitted_table = np.array(result_table)

    plt.show()
    return gradient_fitted_table, shapes

def order_points_to_line(points):
    """
    Sort connected input points to a line.

    Parameters
    ----------
    points: ndarray
        Sort input points (nx2) to a line

    Returns
    -------
    points: ndarray
        Sorted output points
    """
    try:
        if points.shape[1]!=2:
            raise ValueError(f"Wrong pointset dim {points.shape[1]} should be 2")
        clf = NearestNeighbors(2).fit(points)
        G = clf.kneighbors_graph()
        T = nx.from_scipy_sparse_matrix(G)
        order = list(nx.dfs_preorder_nodes(T, 0))
    except:
        order= [0]
        #todo: wright a valid exception !!!!
    return order

@numba.jit(nopython=True)
def get_candidates_accelerated(maximum, dis_transform, image_canny, canny_candidates, threshold):
    """
    Numba accelerated code (precompiles to C++) to calculate possible candidates for line profiling.
    I.e. The values of maximal distance for closed holes in the image.
    Should correspond to the maximum width of the SNC

    Parameters
    ----------
    maximum: float
        Max value of dis_tranform
    dis_transform: ndarray
        Distance transformed floodfill image. Gives the distance of an inclosed pixel to the next edge
    image_canny: ndarray
        Canny processed image
    canny_candidates: ndarray
        Empty array with size image_canny.shape
    threshold: float
        Threshold for a distance value to be accepted as a candidate
    """
    sub_array = np.zeros((2*maximum, 2*maximum))
    for i in range(dis_transform.shape[0]):
        for j in range(dis_transform.shape[1]):
            if dis_transform[i, j] != 0:
                if i+maximum > dis_transform.shape[0] or j+maximum> dis_transform.shape[1] or i-maximum<0 or j-maximum<0:
                    print("out of bounds 1")
                    continue
                for k in range(2*maximum):
                    for l in range(2*maximum):
                        sub_array[k,l] = dis_transform[i - maximum + k, j - maximum+l]
                max_value= sub_array.max()
                for k in range(2*maximum):
                    for l in range(2*maximum):
                        if dis_transform[i - maximum + k, j - maximum+l] < threshold*max_value:
                            dis_transform[i - maximum + k, j - maximum+l] = 0

    # get edges with minimal distance from middle of holes
    # cv2.cvtColor(image_canny,cv2.COLOR_GRAY2RGB)
    for i in range(dis_transform.shape[0]):
        for j in range(dis_transform.shape[1]):
            if dis_transform[i, j] != 0:
                if i+maximum > dis_transform.shape[0] or j+maximum> dis_transform.shape[1] or i-maximum<0 or j-maximum<0:
                    print("out of bounds 2")
                    continue
                for k in range(2*maximum):
                    for l in range(2*maximum):
                        sub_array[k,l] = image_canny[i - maximum + k, j - maximum+l]
                dis_sub = np.ones_like(sub_array).astype(np.float32) * 255
                for k in range(sub_array.shape[0]):
                    for l in range(sub_array.shape[1]):
                        if sub_array[k, l] != 0:
                            dis_sub[k, l] = np.sqrt((k - maximum) ** 2 + (l - maximum) ** 2)
                min_value = dis_sub.min()
                for k in range(sub_array.shape[0]):
                    for l in range(sub_array.shape[1]):
                        if dis_sub[k, l] == min_value:
                            canny_candidates[i - maximum + k, j - maximum+l] = 1

def line_profile(image, start, end, px_size=0.032, sampling=1):
    num = np.linalg.norm(np.array(start) - np.array(end)) * px_size * 100 * sampling
    x, y = np.linspace(start[0], end[0], num), np.linspace(start[1], end[1], num)
    return ndimage.map_coordinates(image, np.vstack((x, y)))


class fit_gaussian:
    """
    Class to perform least_square fitting on a dataset.
    Currently supports one to three gaussian functions.
    """

    def fit_data(self, data, px_size, sampling, nth_line, path, c=(1.0,0.0,0.0,1.0), n_profiles=0):
        """
        Fit given data to one to or three gaussians

        Parameters
        ----------
        data: ndarray
        px_size: float [micro meter]
        sampling: int
        nth_line: int
        path: str
        c: tuble
        n_profiles: int


        """
        matplotlib.rc('font', **{'size' : 12},)
        matplotlib.rcParams['font.sans-serif'] = "Helvetica"

        x = np.linspace(0, data.shape[0]-1, data.shape[0])
        x_aligned = x-30 * px_size * 100 * sampling+(50*px_size*100)

        optim = self.fit_data_to_gaussian(data)
        #plot fit
        fig = plt.figure()
        ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        ax1.plot(x_aligned, self.gaussian(x, optim[0],optim[1],optim[2],optim[-1])/data.max(),
                 lw=1, c='r', ls='--', label='bi-Gaussian fit')
        ax1.plot(x_aligned, self.gaussian(x, optim[3], optim[4], optim[5], optim[-1])/data.max(),
                 lw=1, c='r', ls='--', )
        #plot data
        ax1.plot(x_aligned, data/data.max(), c=c, label="averaged line profile")
        ax1.legend(loc='best')
        ax1.set_ylabel("normed intensity [a.u.]")
        ax1.set_xlabel("distance [nm]")# coordinate space perpendicular to spline fit

        optim_print = np.around(optim, decimals=2)
        txt = f"Bi-gaussian fit parameters: \n" \
              f"Number of profiles: {n_profiles} \n" \
              f"Peak distance: {np.abs(optim_print[2]-optim_print[5]):.2f} \n" \
              f"Width: {optim_print[1]:.2f}, {optim_print[4]:.2f} \n" \
              f"Intensity: {optim_print[0]:.2f}, {optim_print[3]:.2f}"
        fig.text(0.5, 0.01, txt, ha='center')
        fig.set_size_inches(7, 8, forward=True)
        #print("distance = ", optim[2]-optim[5])
        #print("offset = ", optim[-1])


        plt.savefig(path +rf'\profile_{nth_line}.png')
        plt.close(fig)
        #plt.show()

    def two_gaussians(self, x, h1, w1, c1, h2, w2, c2, noise_lvl):
        """
        Double gaussian function.

        Parameters
        ----------
        See :func: `gaussian`


        Returns
        -------
        gaussian: ndarray
            (nx1) array of y values corresponding to the given parameters
        """
        return (self.gaussian(x, h1, w1, c1, 0) +
                self.gaussian(x, h2, w2, c2, 0) + noise_lvl)

    def three_gaussians(self, x, h1, w1, c1, h2, w2, c2, h3, w3, c3, noise_lvl):
        """
        Triple gaussian function.

        Parameters
        ----------
        See :func: `gaussian`


        Returns
        -------
        gaussian: ndarray
            (nx1) array of y values corresponding to the given parameters
        """
        return (self.gaussian(x, h1, w1, c1, 0) +
                self.gaussian(x, h2, w2, c2, 0) +
                self.gaussian(x, h3, w3, c3, 0) + noise_lvl)

    def fit_data_to_gaussian(self, data):
        """
        Fit one, two and three gaussians to given data per least square optimization. Compute and  print chi2.
        Return the optimal parameters found for two gaussians.

        Parameters
        ----------
        data: ndarray
            Given data (1d)

        Returns
        -------
        optim2: tuple
            Optimal parameters to fit two gaussians to data
        """
        x = np.linspace(0, data.shape[0]-1, data.shape[0])
        maximas = self.find_maximas(data)
        print(maximas)
        height = data.max()/2
        guess = [height, 0.5, maximas[0], 0]
        bounds = np.array([[0,data.max()+0.1],[0,np.inf],[0,600],
                            [0,0.1]]).T
        guess2 = [height, 0.5, maximas[0],
                  height, 0.5, maximas[1],0]
        bounds2 = np.array([[0,data.max()+0.1],[0,np.inf],[0,600],
                            [0, data.max()+0.1], [0, np.inf], [0,600],[0,0.1]]).T
        guess3 = [height, 0.5, maximas[0],
                  height, 0.5, maximas[0],
                  height, 0.5, maximas[0], 0]
        bounds3 = np.array([[0,data.max()+0.1],[0,np.inf],[0,600],
                            [0, data.max()+0.1], [0, np.inf], [0,600],
                            [0, data.max()+0.1], [0, np.inf], [0, 600],[0,0.1]]).T

        # calculate error by squared distance to data
        errfunc = lambda p, x, y: (self.gaussian(x, *p) - y) ** 2
        errfunc2 = lambda p, x, y: (self.two_gaussians(x, *p) - y) ** 2
        errfunc3 = lambda p, x, y: (self.three_gaussians(x, *p) - y) ** 2

        result = optimize.least_squares(errfunc, guess[:], bounds=bounds, args=(x, data))
        optim = result.x
        result2 = optimize.least_squares(errfunc2, guess2[:], bounds=bounds2, args=(x, data))
        optim2 = result2.x
        result3 = optimize.least_squares(errfunc3, guess3[:], bounds=bounds3, args=(x, data))
        optim3 = result3.x


        chi1 = lambda p, x, y: ((self.gaussian(x, *p) - y) ** 2)/self.gaussian(x, *p)
        chi2 = lambda p, x, y: ((self.two_gaussians(x, *p) - y) ** 2)/self.two_gaussians(x, *p)
        chi3 = lambda p, x, y: ((self.three_gaussians(x, *p) - y) ** 2)/self.three_gaussians(x, *p)

        err = chi1(optim, x, data).sum()
        err2 = chi2(optim2, x, data).sum()
        err3 = chi3(optim3, x, data).sum()

        print(f"one gaussian chi2 {err}, cost {result.cost} \ntwo gaussian chi2 {err2}, cost {result2.cost} \nthree gaussian chi2 {err3}, cost {result3.cost}")
        print(f"gaussian width {int(optim2[1])}, {int(optim2[4])}")
        print(optim2)
        return optim2

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

    @staticmethod
    def gaussian(x, height, width, center, noise_lvl):
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