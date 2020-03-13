import os

from PyQt5.QtWidgets import *

import numpy as np

from scipy import misc

import czifile

import tifffile

from lxml import etree as XMLET

from PyQt5.QtCore import QAbstractListModel, QModelIndex, QVariant, QAbstractTableModel
from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtCore import Qt


class CustomRowWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super(CustomRowWidget, self).__init__(*args[1:], **kwargs)
        self.setAcceptDrops(True)
        self.file_path = str(args[0])
        self.isParsingNeeded = True
        self.z_file_path = None
        self.extensions = ["czi", "tiff", "tif", "lsm", "png"]
        self.row = QHBoxLayout()
        self.row.addWidget(QLabel(args[0].split(os.sep)[-1]))
        self.pushButtonOpenZ = QPushButton()
        self.pushButtonOpenZ.setText("Open z-stack")
        self.pushButtonOpenZ.clicked.connect(self.open_z)
        self.row.addWidget(self.pushButtonOpenZ)
        self.setLayout(self.row)

    def open_z(self):
        file_dialog = QFileDialog()
        title = "Open z-stack"
        # extensions = "Confocal images (*.jpg; *.png; *.tif;);;Confocal stacks (*.ics)"
        # extensions = "Confocal images (*.jpg *.png *.tif *.ics)"
        extensions = "image (*.czi *.tiff *.tif *.lsm *.png" \
                     ")"
        files_list = QFileDialog.getOpenFileNames(file_dialog, title,
                                                            os.getcwd(), extensions)[0]
        self.z_file_path = files_list[0]
        self.row.removeWidget(self.pushButtonOpenZ)
        self.row.addWidget(QLabel(self.z_file_path))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            if all([str(url.toLocalFile()).split(".")[-1] in self.extensions for url in e.mimeData().urls()]):
                e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            if all([str(url.toLocalFile()).split(".")[-1] in self.extensions for url in e.mimeData().urls()]):
                e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        """
        Drop files directly onto the widget
        File locations are stored in fname
        :param e:
        :return:
        """
        if e.mimeData().hasUrls:
            e.setDropAction(Qt.CopyAction)
            e.accept()
            # Workaround for OSx dragging and dropping
            for url in e.mimeData().urls():
                self.z_file_path = str(url.toLocalFile())
                self.row.removeWidget(self.pushButtonOpenZ)
                self.row.addWidget(QLabel(self.z_file_path))
        else:
            e.ignore()


#Super class for file handling via widget
class MicroscopeImage(QListWidgetItem):
    def __init__(self, *args, **kwargs):
        super(MicroscopeImage, self).__init__(*args, **kwargs)
        self.setText(args[0].split(os.sep)[-1])
        self.file_path = str(args[0])
        self.isParsingNeeded = True
        # self.channels = {}


#Rewritten class to read different formats of microscope images and bring them in the same shape.
#Supportet formats are .czi; .lsm ;.tiff
#Tiff images will most likely not contain the needed metadata -> Pixel size must be set manually.
#Batch mechanism off Zeiss SIM will result in broken header file and is not supported yet.
#Output arrays will be reshaped to [Color[ZStack[X[Y]]]].
#See MicroscopeImage for input.

class ImageSIM(CustomRowWidget):
    """ImageSIM is an instance of QListWidget class, used to read and order SIM data from common file formats.
    Supported formats are: .czi, .lsm, .tif
    """
    #Initialisation see class MicroscopeImage.
    def __init__(self, *args, **kwargs):
        super(ImageSIM, self).__init__(*args, **kwargs)
        self.reset_data()

    #Reset ConfocalImage attributes.
    def reset_data(self):
        self.data = []
        self.relevantData = []
        self.metaData = {}
        self._index = np.zeros(4).astype(np.uint8)
        self.isParsingNeeded = True
        self.extend = None
        self._flip = {}
        self._channel = np.zeros(4).astype(np.bool)
        self.data_z = None

    #Read the image data and metadata und give them into a numpy array.
    #Rearrange the arrays into a consistent shape.
    def parse(self, calibration_px=0.0322, ApplyButton=False):
        self.isParsingNeeded = False
        self.metaData = {}
        self.data_z = None
        self.data = []
        self.Shape = np.ones(1,dtype={'names':["SizeX","SizeY","SizeZ","SizeC"],'formats':['i4','i4','i4','i4']})
        self.extend = os.path.splitext(self.file_path)[1]
        self._color = np.array(([[1,0,0,1],[0,1,0,1],[0,0,1,1],[1,1,0,1]]))
        #CZI files
        if self.extend == '.czi':
            with czifile.CziFile(self.file_path) as czi:
                self.data = czi.asarray()
                #Get relevant part of file header => Metadata.
                Header_Metadata = czi.metadata#str(czi.decode("utf-8")).split('<ImageDocument>')
                Metadata = XMLET.fromstring(Header_Metadata)
                try:
                    #Query XML fore the metadata for picture shape(X;Y;Z-stacks).
                    #Picture Shape.
                    shapes = Metadata.findall('./Metadata/Information/Image')[0]
                    self.metaData["ShapeSizeX"] = int(shapes.findall('SizeX')[0].text)
                    self.metaData["ShapeSizeY"] = int(shapes.findall('SizeY')[0].text)
                    try:
                        self.metaData["ShapeSizeZ"] = int(shapes.findall('SizeZ')[0].text)
                    except:
                        self.metaData["ShapeSizeZ"] = 1
                        print("One z-Slice")
                    #Get the hyperstack dimension if the image is a hyperstack.
                    try:
                        self.metaData["ShapeSizeC"] = int(shapes.findall('SizeC')[0].text)
                    except:
                        self.metaData["ShapeSizeC"] = 1
                        print("One Channel")
                    #Get physical pixel size of image(nm/px) convert to(µm/px).
                    PixelSizes = Metadata.findall('./Metadata/Scaling/Items/Distance')
                    self.metaData['SizeX'] = float(PixelSizes[0].findall('Value')[0].text)*10**6
                    self.metaData['SizeY'] = float(PixelSizes[1].findall('Value')[0].text)*10**6
                    self.metaData['SizeZ'] = float(PixelSizes[2].findall('Value')[0].text)*10**6
                except:
                    raise
                    print("Metadata fail")

        #Tiff files.
        #Tiff files are problematic because they most likely wont contain the nessecary metadata.
        #Try to get the shape info over common dimensions.
        elif self.extend == '.tif' or self.extend == '.tiff':
            #z_name = os.path.splitext(self.file_path)[0]+"-z-stack.tif"
            if self.z_file_path is not None:
                if os.path.exists(self.z_file_path):
                    with tifffile.TiffFile(self.z_file_path) as tif:
                        self.data_z = tif.asarray()
                #z_name = os.path.splitext(self.file_path)[0]+"-z-stack.tiff"
            #if os.path.exists(z_name):
            #    with tifffile.TiffFile(z_name) as tif:
            #        self.data_z = tif.asarray()
            with tifffile.TiffFile(self.file_path) as tif:
                #print(tif.imagej_metadata)
                self.data = tif.asarray()#[...,0]#np.moveaxis(tif.asarray(),0,1)

                #self.data = np.rollaxis(self.data,0,1)
                #self.data = np.rollaxis(self.data,2,0)
                self.metaData["ShapeSizeC"] = 3
                self.metaData["ShapeSizeZ"] = 1
                self.metaData["SizeZ"] = 1
                self.metaData["SizeX"] = calibration_px
                self.metaData["SizeY"] = calibration_px
                self.metaData["ShapeSizeY"] = self.data.shape[-2]
                self.metaData["ShapeSizeX"] = self.data.shape[-1]
                for page in tif.pages:
                    for tag in page.tags.values():
                        tag_name, tag_value = tag.name, tag.value
                        #print(tag_name, tag_value)
                        if "ImageDescription" in tag_name:
                            tags = tag_value.split("\n")
                            axes = []
                            lengths=[]
                            for tag in tags:
                                if "axes" in tag:
                                    axes = tag.split("=")[-1].split(",")
                                    print("calculating axe dimensions")
                                if "lengths" in tag:
                                    lengths = tag.split("=")[-1].split(",")
                                if "slices" in tag:
                                    print("Found Z Stack")
                                    axes.append("Slices")
                                    lengths.append(tag.split("=")[-1])
                                if "channels" in tag:
                                    print("Found Color Channels")
                                    axes.append("Channels")
                                    lengths.append(tag.split("=")[-1])
                # for i,axe in enumerate(axes):
                #     if "X" in axe:
                #         self.metaData["ShapeSizeX"] = int(lengths[i])
                #     if "Y" in axe:
                #         self.metaData["ShapeSizeX"] = int(lengths[i])
                #     if "Channel" in axe:
                #         self.metaData["ShapeSizeC"] = int(lengths[i])
                #     if "Slices" in axe:
                #         self.metaData["ShapeSizeZ"] = int(lengths[i])


        #Read Lsm Files.
        elif self.extend == '.lsm':
            with tifffile.TiffFile(self.file_path) as tif:
                self.data = tif.asarray(memmap=True)
                headerMetadata = str(tif.pages[0].cz_lsm_scan_info)
                metadataList = headerMetadata.split("\n*")
                #Get image shape from lsm header SizeC=0 if not given.
                for shapes in metadataList:
                    if "images_height" in shapes:
                        self.metaData["ShapeSizeX"]= int(shapes.split()[-1])
                    if "images_width" in shapes:
                        self.metaData["ShapeSizeY"]= int(shapes.split()[-1])
                    if "images_number_planes" in shapes:
                        self.metaData["ShapeSizeZ"]= int(shapes.split()[-1])
                    if "images_number_channels" in shapes:
                        self.metaData["ShapeSizeC"]= int(shapes.split()[-1])
                #Get physical pixel size of image(nm/px) convert to(µm/px).
                self.data = np.swapaxes(self.data,1,2)
                LsmPixelHeader = str(tif.pages[0].tags.cz_lsm_info)
                LsmInfo = LsmPixelHeader.split(", ")
                i = 0
                #Query for pixel size.
                for element in LsmInfo:

                    if "e-0" in element:
                        i += 1
                        if i == 1:
                            self.metaData['SizeX'] = (float(element)*10**6)
                        if i == 2:
                            self.metaData['SizeY'] = (float(element)*10**6)
                        if i == 3:
                            self.metaData['SizeZ'] = (float(element)*10**6)

        elif self.extend == ".png":
            self.data = misc.imread(self.file_path)
            self.data = np.expand_dims(np.expand_dims(self.data[...,0],0),0)
            self.metaData["ShapeSizeC"] = 1
            self.metaData["ShapeSizeZ"] = 1
            self.metaData["ShapeSizeX"] = self.data.shape[2]
            self.metaData["ShapeSizeY"] = self.data.shape[3]
            self.metaData["SizeZ"] = 1
            self.metaData["SizeX"] = 0.01
            self.metaData["SizeY"] = 0.01
        #todo:hack 2d into 4d
        if len(self.data.shape) ==2:
            new_data = np.zeros((3,1,self.data.shape[0], self.data.shape[1]))
            self.metaData['SizeX'] = calibration_px
            self.metaData['SizeY'] = calibration_px

            new_data[0,0] = self.data
            self.data = new_data
        if len(self.data.shape) == 3:
            if self.data.shape[-1] <5:
                new_data = np.zeros((3,1,self.data.shape[0],self.data.shape[1]))
                new_data[0,:] = np.sum(self.data,axis=2)
                self.metaData["ShapeSizeY"] = self.data.shape[0]
                self.metaData["ShapeSizeX"] = self.data.shape[1]
            else:
                new_data = np.zeros((3,1,self.data.shape[-2], self.data.shape[-1]))
                self.metaData['SizeX'] = calibration_px
                self.metaData['SizeY'] = calibration_px

                for i in range(self.data.shape[0]):
                    new_data[i,0] = self.data[i]
            self.data = new_data
        # #Bring all formats in the same shape.
        for i,n in enumerate(self.data.shape):
            if n == self.metaData["ShapeSizeC"]:
                self.data = np.rollaxis(self.data, i, 0)
            if n == self.metaData["ShapeSizeZ"]:
                self.data = np.rollaxis(self.data, i, 1)
            if n == self.metaData["ShapeSizeY"]:
                self.data = np.rollaxis(self.data, i, 2)
            if n == self.metaData["ShapeSizeX"]:
                self.data = np.rollaxis(self.data, i, 3)
        self.data = np.reshape(self.data,(self.metaData["ShapeSizeC"],self.metaData["ShapeSizeZ"],self.metaData["ShapeSizeY"],self.metaData["ShapeSizeX"]))
        self.metaData['ChannelNum'] = self.metaData["ShapeSizeC"]
        #Set pixel size to manuell value if there are no metadata.
        if self.metaData == {}:
            self.set_calibration(calibration_px)
        #Set the box for manuel calibration to the actuell pixel size.


    @property
    def data_rgba_2d(self):
        if not np.any(self._channel):
            raise ValueError("No channel visible")
        visible_data = self.data[(np.where(self._channel))]
        data_rgba = np.zeros((visible_data.shape[0], self.metaData["ShapeSizeY"],self.metaData["ShapeSizeX"],4))
        indices = self._index[np.where(self._channel)]
        for i in range(visible_data.shape[0]):
            data_rgba[i] = np.stack((visible_data[i,indices[i]],)*4,axis=-1)
            data_rgba[i] *= self._color[np.where(self._channel)][i]
        return data_rgba

    @property
    def data_gray_2d(self):
        if not np.any(self._channel):
            raise ValueError("No channel visible")
        visible_data = self.data[(np.where(self._channel))]
        data_gray = np.zeros((visible_data.shape[0], self.metaData["ShapeSizeY"],self.metaData["ShapeSizeX"]))
        indices = self._index[np.where(self._channel)]
        for i in range(visible_data.shape[0]):
            data_gray[i] = visible_data[i,indices[i]]
        return data_gray


    @property
    def data_rgba_3d(self):
        return self.data[np.where(self._channel)]

    @property
    def channel(self, index):
        return self._channel[index]

    @channel.setter
    def channel(self, value):
        self._channel[value[0]] = value[1]

    @property
    def index(self, channel):
        return self._index[channel]

    @index.setter
    def index(self, value):
        if value[1] > self.metaData["ShapeSizeZ"]:
            raise ValueError("Index out of bounds")
        self._index[value[0]] = value[1]

    @property
    def color(self, channel):
        return self._color[channel]

    @color.setter
    def color(self, channel, value):
        if value.shape[0] != 4:
            raise ValueError("Not a color")
        self._color[channel] = value

    @property
    def flip(self, direction):
        """
        directions: UpsideDown, LeftRight
        value: True, False
        """
        return self._flip[direction]

    @flip.setter
    def flip(self, direction, value):
        self._flip[direction] = value

    #Set pixel size to manuell value.
    def set_calibration(self, px):
        self.metaData['SizeX'] = px
        self.metaData['SizeY'] = px
        self.metaData['SizeZ'] = px