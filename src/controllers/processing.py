import numpy as np
from PyQt5.QtCore import QThread,pyqtSignal
import os
from matplotlib import cm


class QSuperThread(QThread):
    """
    Processing thread to compute distances between SNCs in a given SIM image.
    Extending the QThread class keeps the GUI running while the evaluation runs in the background.
    """
    sig = pyqtSignal(int)
    sig_plot_data = pyqtSignal(np.ndarray, int, int, str, tuple, int)
    done = pyqtSignal(int)

    def __init__(self, parent=None):
        super(QSuperThread, self).__init__(parent)
        self.z_project_collection = False
        self._blur = 20
        self._intensity_threshold = 1
        self.colormap = cm.gist_ncar
        self.sampling = 10
        self._spline_parameter = 1.0
        self.image_stack = None
        self._data_z = None


    def set_data(self,ID, image_stack, f_name):
        """
        Set data for processing

        Parameters
        ----------
        image_stack: ndarray
            Ndimage of shape (C,Z,X,Y)
        px_size: float
            Pixel size [micro meter]
        f_name: str
            File name to save processed data to.

        """
        self.ID = ID
        path = os.path.dirname(os.getcwd()) + r"\data\\" + os.path.splitext(os.path.basename(f_name))[0]
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.image_stack = image_stack[0:3]

    @property
    def intensity_threshold(self):
        return self._intensity_threshold

    @intensity_threshold.setter
    def intensity_threshold(self, value):
        self._intensity_threshold = np.exp(value)

    @property
    def px_size(self):
        return self._px_size

    @px_size.setter
    def px_size(self, value):
        self._px_size = value

    @property
    def blur(self):
        return self._blur

    @blur.setter
    def blur(self, value):
        self._blur = value

    @property
    def spline_parameter(self):
        return self._spline_parameter

    @spline_parameter.setter
    def spline_parameter(self, value):
        self._spline_parameter = value

    @property
    def data_z(self):
        return self._data_z

    @data_z.setter
    def data_z(self, data):
        self.z_project_collection = True
        self._data_z = data

