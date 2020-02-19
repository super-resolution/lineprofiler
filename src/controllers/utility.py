"""
Utility package for image processing featuring funtcions like:
 * Create gradient images
 * Compute the orientation and position line resembling patterns in an image.
 * Fit functions to a one dimensional dataset and save the plot in a given path
"""
import numba
import cv2
import numpy as np
from skimage.morphology import skeletonize_3d
from skimage.measure import label
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema
import networkx as nx

from scipy import ndimage

def line_parameters(point, direction):
    """

    Parameters
    ----------
    point: tuple
        starting point for line
    direction: float
        direction of line


    Returns
    -------
    result: dict
        Containing arrays with X and Y coordinates of line as well as start and endpoint
    """
    x_i = -40 * np.cos(direction)
    y_i = -40 * np.sin(direction)
    result = dict()
    result['start'] = [point[0] - x_i, point[1] - y_i]
    result['end'] = [point[0] + x_i, point[1] + y_i]
    # length of line
    num = int(np.round(np.sqrt(x_i ** 2 + y_i ** 2)))
    result['X'] = np.linspace(point[0] - x_i, point[0] + x_i, 3 * num)
    result['Y'] = np.linspace(point[1] - y_i, point[1] + y_i, 3 * num)

    return result


def create_floodfill_image(image):
    """
    .. _floodfill:

    Create a floodfill image applying a border with zeros around the image. This results in every boundary
    point, being a source point for the floodfill algorithm.

    Parameters
    ----------
    image: ndarray
        2D input image

    Returns
    -------
    floodfill image: ndarray
        2D binary output image
    """

    ret, thresh = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    # make black border sorounding the image
    thresh = cv2.copyMakeBorder(thresh,1,1,1,1,cv2.BORDER_CONSTANT)

    # floodfill image to get interior forms
    mask = np.zeros((thresh.shape[0] + 2, thresh.shape[1] + 2), np.uint8)
    cv2.floodFill(thresh, mask, (0, 0), 255)
    cv2.imwrite(r"C:\Users\biophys\Documents\Klosters\flood.tif", cv2.bitwise_not(thresh[1:thresh.shape[0]-1,1:thresh.shape[1]-1]))

    # discard border in returned image
    return cv2.bitwise_not(thresh[1:thresh.shape[0]-1,1:thresh.shape[1]-1])


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
            if maxima_value[index] < 0.7 * maximum:
                values[i] = values[0]
                continue

            maximum = maxima_value[index]
            maxima_value[index] = 0

            values[i] = maximas[0][index]
        except:
            print("zero exception")
    return values

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
        Array of the pixel orientation in a box of "sobel" size (unit = rad)

    Example
    -------
    >>> image = cv2.imread("path_to_file.png")
    >>> gradient_image = create_gradient_image(image, 3)
    """

    image_b = cv2.blur(image, (blur, blur))
    X = cv2.Sobel(image_b.astype(np.float64), cv2.CV_64F, 1, 0, ksize=sobel)
    Y = cv2.Sobel(image_b.astype(np.float64), cv2.CV_64F, 0, 1, ksize=sobel)
    return np.arctan2(X, Y)


def compute_line_orientation(image, blur, min_len=100, spline=3, expansion=1, expansion2=1):
    """
    Compute the orientation and position of line resembling patterns in an image.

    The image is convolved with a gaussian blur compensating for noise discontinuity or holes.
    A thresholding algorithm (1) converts the image from grayscale to binary. Using Lees algorithm (2)
    the expanded lines are reduced to one pixel width. The pixel coordinates from all still connected lines
    are retrieved and tested for continuity. Points of discontinuity are used as breakpoints and all following
    coordinates connected to a new line. Lines, shorter than the minimum required length are discarted.
    An univariate spline of degree 3 is fitted to each line. Note that shape and gradient of the line depend on the
    smoothing parameter. The rounded coordinates and their derivatives are returned in a table,
    together with the length of each line.

    Parameters
    ----------
    image: ndarray
        Image containing line resembling patterns
    blur: int
        Amount of blur to apply to the image. Should be in the order of magnitude of the line width in pixel.
    min_len: int
        Minimal accepted line length
    smooth: float
        Positive smoothing factor used to choose the number of knots
    spline: int
        Degree of the smoothing spline. Must be <= 5. Default is 3, a cubic spline.

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
    >>> import matplotlib.pyplot as plt
    >>> from src.controllers.utility import *
    >>> import tifffile as tif
    >>> import os
    >>>
    >>> with tif.TiffFile(os.path.dirname(os.getcwd()) + r"\test_data_microtub\Expansion dSTORM-Line Profile test.tif") as file:
    >>>     image = file.asarray().astype(np.uint8)*40
    >>>
    >>> fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharey=True)
    >>> axs[0].imshow(image)
    >>> axs[0].set_xlabel("test_image")
    >>> axs[1].imshow(image)
    >>> axs[1].set_xlabel("test_image with fitted splines")
    >>> spline_table, shapes = compute_line_orientation(image, 20)
    >>> spline_positions = spline_table[:,0:2]
    >>> index = 0
    >>> for i in range(len(shapes)):
    >>>     axs[1].plot(spline_positions[index:index+shapes[i],1],spline_positions[index:index+shapes[i],0], c="r")
    >>>     index += shapes[i]
    >>> plt.show()

    .. figure:: fig/spline_fitting.png

    """
    image = cv2.blur(image, (blur, blur))

    # build threshold image
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)

    skeleton = skeletonize_3d((thresh / 255).astype(np.uint8)).astype(np.uint8)

    cv2.imwrite(r"C:\Users\biophys\Documents\Klosters\skel.tif", skeleton)
    # contour = self.collapse_contours(contours)
    #cv2.imshow("asdf", skeleton * 255)
    #cv2.waitKey(0)

    colormap = label(skeleton, connectivity=2)
    lines = []
    for i in range(colormap.max()):
        j = i + 1
        line = np.where(colormap == j)
        if len(line[0]) > min_len:
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
            if i + 30 < distance.shape[0]:
                vec1 = points[i + 15] - points[i]
                vec2 = points[i + 30] - points[i + 15]
                direction = np.dot(vec1, vec2)
                if direction < 30:
                    direction_change = i +1
            if distance[i] - distance[i - 1] > 10 or i > direction_change:
                distance = distance[:i]
                lines.append(points[i + 2:].T)
                points = points[:i + 1]
                line_length += 1
                break
        if points.shape[0] < min_len:
            continue
        distance = np.insert(distance, 0, 0) / distance[-1]
        point_list.append(points)

        # Build a list of the spline function, one for each dimension:
        smooth = points.shape[0]
        splines = [UnivariateSpline(distance, coords, k=spline, s=smooth*expansion) for coords in points.T]
        dsplines = [spline.derivative() for spline in splines]
        splines = [UnivariateSpline(distance, coords, k=spline, s=smooth*expansion2) for coords in points.T]
        # Computed the spline for the asked distances:
        alpha = np.linspace(0, 1, points.shape[0])
        points_fitted = np.vstack(spl(alpha) for spl in splines).T
        #append fitted points to
        point_fitted_list.append(points_fitted)
        gradient_list.append(np.vstack(spl(alpha) for spl in dsplines).T)

        #plot results for testing purposes
        #plt.plot(points_fitted[..., 1], points_fitted[..., 0])

    #sort results to array
    result_table = []
    shapes = []
    for i in range(len(point_fitted_list)):
        shapes.append(point_fitted_list[i].shape[0])
        for j in range(point_fitted_list[i].shape[0]):
            result_table.append([int(point_fitted_list[i][j][0]), int(point_fitted_list[i][j][1]),
                                 gradient_list[i][j][0], gradient_list[i][j][1],int(point_list[i][j][0]), int(point_list[i][j][1])], )
    gradient_fitted_table = np.array(result_table)




    return gradient_fitted_table, shapes

def order_points_to_line(points):
    """
    Determine the two nearest neighbors of each input point. Write the connectivity in a sparse matrix.
    Determine the order of the points.

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
    num = int(np.round(np.linalg.norm(np.array(start) - np.array(end)) * px_size * 100 * sampling))
    x, y = np.linspace(start[0], end[0], num), np.linspace(start[1], end[1], num)
    return ndimage.map_coordinates(image, np.vstack((x, y)))

def calc_peak_distance(profile):
    split1 = profile[:int(profile.shape[0]/2)]
    split2 = profile[int(profile.shape[0]/2):]
    distance= (split2.argmax() + profile.shape[0]/2) - split1.argmax()
    center = split1.argmax() + distance/2
    return distance, center

