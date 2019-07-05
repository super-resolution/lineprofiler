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
from scipy.stats import chisquare

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
    Create distance transform of closed shapes in the current self.image
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
        if len(line[0]) > 30:
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
        if points.shape[0] < 10:
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
                if direction < 25:
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
    for i in range(len(point_fitted_list)):
        for j in range(point_fitted_list[i].shape[0]):
            result_table.append([int(point_fitted_list[i][j][0]), int(point_fitted_list[i][j][1]),
                                 gradient_list[i][j][0], gradient_list[i][j][1]])
    gradient_fitted_table = np.array(result_table)

    plt.show()
    return gradient_fitted_table

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


def gaussian(x, height, center, width, noise_lvl):
    return height * np.exp(-(x - center) ** 2 / (2 * width ** 2)) + noise_lvl

def two_gaussians(x, h1, c1, w1, h2, c2, w2, noise_lvl):
    return (gaussian(x, h1, c1, w1, 0) +
            gaussian(x, h2, c2, w2, 0) + noise_lvl)

def three_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, noise_lvl):
    return (gaussian(x, h1, c1, w1, 0) +
            gaussian(x, h2, c2, w2, 0) +
            gaussian(x, h3, c3, w3, 0) + noise_lvl)

def find_maximas(data, n=3):
    maximas = argrelextrema(data, np.greater)
    maxima_value = data[maximas]
    values = np.zeros(n)
    for i in range(n):
        index = np.argmax(maxima_value)
        maxima_value[index] = 0
        values[i] = maximas[0][index]
    return values


def fit_data_to_gaussian(data):
    x = np.linspace(0, data.shape[0]-1, data.shape[0])
    maximas = find_maximas(data)
    print(maximas)
    height = 0.5
    guess = [height, 0.5, maximas[0], 0]
    guess2 = [height, 0.5, maximas[0],
              height, 0.5, maximas[1], 0]
    guess3 = [height, 0.5, maximas[0],
              height, 0.5, maximas[1],
              height, 0.5, maximas[2], 0]

    # calculate error by squared distance to data
    errfunc = lambda p, x, y: (gaussian(x, *p) - y) ** 2
    errfunc2 = lambda p, x, y: (two_gaussians(x, *p) - y) ** 2
    errfunc3 = lambda p, x, y: (three_gaussians(x, *p) - y) ** 2

    optim, success = optimize.leastsq(errfunc, guess[:], args=(x, data))
    optim2, success = optimize.leastsq(errfunc2, guess2[:], args=(x, data))
    optim3, success = optimize.leastsq(errfunc3, guess3[:], args=(x, data))


    chi1 = lambda p, x, y: ((gaussian(x, *p) - y) ** 2)/y
    chi2 = lambda p, x, y: ((two_gaussians(x, *p) - y) ** 2)/y
    chi3 = lambda p, x, y: ((three_gaussians(x, *p) - y) ** 2)/y

    err = chi1(optim, x, data).sum()
    err2 = chi2(optim2, x, data).sum()
    err3 = chi3(optim3, x, data).sum()

    print(f"chi2 one gaussian {err} \nchi2 two gaussian {err2} \nchi3 three gaussian {err3}")

    return optim2

