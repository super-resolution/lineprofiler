# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src\gui\random.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(951, 735)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/icons/molecules2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("QWidget{\n"
"    background-color: rgb(67, 67, 67);\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QScrollArea{\n"
"    border: 0px;\n"
"    margin: 0px;\n"
"    padding: 0px;\n"
"}\n"
"\n"
"QToolTip { \n"
"    color: #000;\n"
"}\n"
"QTabBar::tab {\n"
"    border: 1px #fff;\n"
"    border-top-style :solid;\n"
"    border-right-style :solid; \n"
"    border-bottom-style :none; \n"
"    border-left-style :solid;\n"
"    padding: 4px 6px;\n"
"    margin: 0px 4px;\n"
"/*\n"
"    border-top-left-radius: 5px;\n"
"    border-top-right-radius: 5px;\n"
"    min-width: 90px;\n"
"    min-height: 75px;\n"
"*/    \n"
"}\n"
"\n"
"QTabBar::tab:selected{\n"
"    background-color: #000000;\n"
"    border: 1px blue;\n"
"    border-top-style :solid;\n"
"    border-right-style :solid; \n"
"    border-bottom-style :none; \n"
"    border-left-style :solid;\n"
"    padding: 4px 6px;\n"
"}\n"
"QTabBar::tab:!selected {\n"
"    margin-top: 4px; }\n"
"/*QGroupBox:title{\n"
"    color: #ffaa00;\n"
"}*/\n"
"\n"
"QGroupBox{\n"
"    color: #ffaa00;\n"
"    \n"
"}\n"
"QPushButton:disabled,\n"
"QPushButton[disabled]{\n"
"    background-color: rgb(50, 50, 50);\n"
"}\n"
"QProgressBar {\n"
"text-align: center;\n"
"padding: -1px;\n"
"border-bottom-right-radius: 10px;\n"
"border-bottom-left-radius: 10px;\n"
"background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0,\n"
"stop: 0 #fff,\n"
"stop: 1 #999999 );\n"
"width: 15px;\n"
"}\n"
"QProgressBar::chunk {\n"
"background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0,\n"
"stop: 0 #006d1b,\n"
"stop: 1 #01a82a );\n"
"\n"
"border-bottom-right-radius: 10px;\n"
"border-bottom-left-radius: 10px;\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(0, 235))
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.frame_images_actions = QtWidgets.QFrame(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.frame_images_actions.setFont(font)
        self.frame_images_actions.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_images_actions.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_images_actions.setLineWidth(1)
        self.frame_images_actions.setObjectName("frame_images_actions")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_images_actions)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_top_controls = QtWidgets.QGroupBox(self.frame_images_actions)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_top_controls.setFont(font)
        self.groupBox_top_controls.setObjectName("groupBox_top_controls")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_top_controls)
        self.verticalLayout_6.setContentsMargins(1, 0, 1, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.tab_5 = QtWidgets.QTabWidget(self.groupBox_top_controls)
        self.tab_5.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(100)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab_5.sizePolicy().hasHeightForWidth())
        self.tab_5.setSizePolicy(sizePolicy)
        self.tab_5.setTabPosition(QtWidgets.QTabWidget.North)
        self.tab_5.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tab_5.setIconSize(QtCore.QSize(20, 20))
        self.tab_5.setDocumentMode(True)
        self.tab_5.setObjectName("tab_5")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.pushButton_open = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_open.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_open.setObjectName("pushButton_open")
        self.horizontalLayout_9.addWidget(self.pushButton_open)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem)
        self.pushButton_close_all = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_close_all.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_close_all.setObjectName("pushButton_close_all")
        self.horizontalLayout_9.addWidget(self.pushButton_close_all)
        self.verticalLayout_2.addLayout(self.horizontalLayout_9)
        self.image_list = QtWidgets.QListWidget(self.tab_3)
        self.image_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.image_list.setObjectName("image_list")
        self.verticalLayout_2.addWidget(self.image_list)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.pushButton_show = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_show.setObjectName("pushButton_show")
        self.horizontalLayout_7.addWidget(self.pushButton_show)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem1)
        self.pushButton_unload = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_unload.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_unload.setObjectName("pushButton_unload")
        self.horizontalLayout_7.addWidget(self.pushButton_unload)
        self.pushButton_close = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_close.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_close.setObjectName("pushButton_close")
        self.horizontalLayout_7.addWidget(self.pushButton_close)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icon/icons/selection13.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tab_5.addTab(self.tab_3, icon1, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab_4)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox_config = QtWidgets.QGroupBox(self.tab_4)
        self.groupBox_config.setMinimumSize(QtCore.QSize(0, 100))
        self.groupBox_config.setObjectName("groupBox_config")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox_config)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 19, 911, 68))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_distance_threshold = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_distance_threshold.setObjectName("label_distance_threshold")
        self.gridLayout.addWidget(self.label_distance_threshold, 0, 0, 1, 1)
        self.label_upper_lim = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_upper_lim.setWordWrap(False)
        self.label_upper_lim.setObjectName("label_upper_lim")
        self.gridLayout.addWidget(self.label_upper_lim, 1, 2, 1, 1)
        self.spinBox_lower_limit = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_lower_limit.setEnabled(True)
        self.spinBox_lower_limit.setMaximum(1000)
        self.spinBox_lower_limit.setProperty("value", 300)
        self.spinBox_lower_limit.setObjectName("spinBox_lower_limit")
        self.gridLayout.addWidget(self.spinBox_lower_limit, 1, 1, 1, 1)
        self.label_lower_lim = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_lower_lim.setObjectName("label_lower_lim")
        self.gridLayout.addWidget(self.label_lower_lim, 1, 0, 1, 1)
        self.spinBox_upper_limit = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_upper_limit.setEnabled(True)
        self.spinBox_upper_limit.setMaximum(99999)
        self.spinBox_upper_limit.setProperty("value", 1500)
        self.spinBox_upper_limit.setObjectName("spinBox_upper_limit")
        self.gridLayout.addWidget(self.spinBox_upper_limit, 1, 3, 1, 1)
        self.spinBox_gaussian_blur = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox_gaussian_blur.setMaximum(999)
        self.spinBox_gaussian_blur.setSingleStep(1)
        self.spinBox_gaussian_blur.setProperty("value", 20)
        self.spinBox_gaussian_blur.setObjectName("spinBox_gaussian_blur")
        self.gridLayout.addWidget(self.spinBox_gaussian_blur, 0, 3, 1, 1)
        self.label_blur = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_blur.setObjectName("label_blur")
        self.gridLayout.addWidget(self.label_blur, 0, 2, 1, 1)
        self.px_size_label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.px_size_label.setObjectName("px_size_label")
        self.gridLayout.addWidget(self.px_size_label, 2, 0, 1, 1)
        self.spinBox_px_size = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.spinBox_px_size.setDecimals(5)
        self.spinBox_px_size.setProperty("value", 0.032)
        self.spinBox_px_size.setObjectName("spinBox_px_size")
        self.gridLayout.addWidget(self.spinBox_px_size, 2, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 2, 1, 1)
        self.doubleSpinBox_spline_parameter = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.doubleSpinBox_spline_parameter.setMinimum(0.01)
        self.doubleSpinBox_spline_parameter.setMaximum(3.0)
        self.doubleSpinBox_spline_parameter.setSingleStep(0.1)
        self.doubleSpinBox_spline_parameter.setProperty("value", 0.32)
        self.doubleSpinBox_spline_parameter.setObjectName("doubleSpinBox_spline_parameter")
        self.gridLayout.addWidget(self.doubleSpinBox_spline_parameter, 2, 3, 1, 1)
        self.comboBox_operation_mode = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboBox_operation_mode.setObjectName("comboBox_operation_mode")
        self.comboBox_operation_mode.addItem("")
        self.comboBox_operation_mode.addItem("")
        self.comboBox_operation_mode.addItem("")
        self.gridLayout.addWidget(self.comboBox_operation_mode, 0, 1, 1, 1)
        self.verticalLayout_4.addWidget(self.groupBox_config)
        self.groupBox_config_calibration = QtWidgets.QGroupBox(self.tab_4)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_config_calibration.setFont(font)
        self.groupBox_config_calibration.setFlat(True)
        self.groupBox_config_calibration.setObjectName("groupBox_config_calibration")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.groupBox_config_calibration)
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.label = QtWidgets.QLabel(self.groupBox_config_calibration)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.doubleSpinBox_intensity_threshold = QtWidgets.QDoubleSpinBox(self.groupBox_config_calibration)
        self.doubleSpinBox_intensity_threshold.setMinimum(-10.0)
        self.doubleSpinBox_intensity_threshold.setMaximum(10.0)
        self.doubleSpinBox_intensity_threshold.setSingleStep(0.1)
        self.doubleSpinBox_intensity_threshold.setProperty("value", 3.5)
        self.doubleSpinBox_intensity_threshold.setObjectName("doubleSpinBox_intensity_threshold")
        self.horizontalLayout_2.addWidget(self.doubleSpinBox_intensity_threshold)
        self.horizontalSlider_intensity_threshold = QtWidgets.QSlider(self.groupBox_config_calibration)
        self.horizontalSlider_intensity_threshold.setMinimum(-100)
        self.horizontalSlider_intensity_threshold.setMaximum(100)
        self.horizontalSlider_intensity_threshold.setProperty("value", 35)
        self.horizontalSlider_intensity_threshold.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_intensity_threshold.setObjectName("horizontalSlider_intensity_threshold")
        self.horizontalLayout_2.addWidget(self.horizontalSlider_intensity_threshold)
        self.horizontalLayout_12.addLayout(self.horizontalLayout_2)
        self.verticalLayout_4.addWidget(self.groupBox_config_calibration)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icon/icons/settings48.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tab_5.addTab(self.tab_4, icon2, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.plotParameters = QtWidgets.QGroupBox(self.tab)
        self.plotParameters.setGeometry(QtCore.QRect(9, 9, 641, 211))
        self.plotParameters.setObjectName("plotParameters")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.plotParameters)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 20, 621, 181))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem4, 3, 0, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem5, 3, 2, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem6, 3, 1, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem7, 3, 3, 1, 1)
        self.checkBox_gaussian = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.checkBox_gaussian.setChecked(False)
        self.checkBox_gaussian.setObjectName("checkBox_gaussian")
        self.gridLayout_2.addWidget(self.checkBox_gaussian, 0, 0, 1, 1)
        self.checkBox_trigaussian = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.checkBox_trigaussian.setChecked(False)
        self.checkBox_trigaussian.setObjectName("checkBox_trigaussian")
        self.gridLayout_2.addWidget(self.checkBox_trigaussian, 1, 0, 1, 1)
        self.checkBox_multi_cylidner_projection = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.checkBox_multi_cylidner_projection.setChecked(False)
        self.checkBox_multi_cylidner_projection.setObjectName("checkBox_multi_cylidner_projection")
        self.gridLayout_2.addWidget(self.checkBox_multi_cylidner_projection, 2, 0, 1, 1)
        self.checkBox_bigaussian = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.checkBox_bigaussian.setChecked(True)
        self.checkBox_bigaussian.setObjectName("checkBox_bigaussian")
        self.gridLayout_2.addWidget(self.checkBox_bigaussian, 0, 1, 1, 1)
        self.checkBox_cylinder_projection = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.checkBox_cylinder_projection.setChecked(False)
        self.checkBox_cylinder_projection.setObjectName("checkBox_cylinder_projection")
        self.gridLayout_2.addWidget(self.checkBox_cylinder_projection, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 2, 1, 1, 1)
        self.doubleSpinBox_expansion_factor = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        self.doubleSpinBox_expansion_factor.setEnabled(False)
        self.doubleSpinBox_expansion_factor.setMinimum(0.5)
        self.doubleSpinBox_expansion_factor.setSingleStep(0.1)
        self.doubleSpinBox_expansion_factor.setProperty("value", 1.0)
        self.doubleSpinBox_expansion_factor.setObjectName("doubleSpinBox_expansion_factor")
        self.gridLayout_2.addWidget(self.doubleSpinBox_expansion_factor, 2, 2, 1, 1)
        self.tab_5.addTab(self.tab, "")
        self.verticalLayout_6.addWidget(self.tab_5)
        self.horizontalLayout.addWidget(self.groupBox_top_controls)
        self.horizontalLayout_14.addWidget(self.frame_images_actions)
        MainWindow.setCentralWidget(self.centralwidget)
        self.dockWidget = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dockWidget.sizePolicy().hasHeightForWidth())
        self.dockWidget.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.dockWidget.setFont(font)
        self.dockWidget.setFloating(False)
        self.dockWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.dockWidget.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dockWidgetContents.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents.setSizePolicy(sizePolicy)
        self.dockWidgetContents.setMinimumSize(QtCore.QSize(0, 427))
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.viewer_container = QtWidgets.QWidget(self.dockWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.viewer_container.sizePolicy().hasHeightForWidth())
        self.viewer_container.setSizePolicy(sizePolicy)
        self.viewer_container.setObjectName("viewer_container")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.viewer_container)
        self.verticalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.viewer_container_layout = QtWidgets.QVBoxLayout()
        self.viewer_container_layout.setSpacing(6)
        self.viewer_container_layout.setObjectName("viewer_container_layout")
        self.verticalLayout_13.addLayout(self.viewer_container_layout)
        self.verticalLayout_11.addWidget(self.viewer_container)
        self.widget_6 = QtWidgets.QWidget(self.dockWidgetContents)
        self.widget_6.setObjectName("widget_6")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.widget_6)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setSpacing(6)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.line_3 = QtWidgets.QFrame(self.widget_6)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout_8.addWidget(self.line_3)
        self.groupBox_confocal_bottom_controls = QtWidgets.QGroupBox(self.widget_6)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_confocal_bottom_controls.setFont(font)
        self.groupBox_confocal_bottom_controls.setStyleSheet("border-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));")
        self.groupBox_confocal_bottom_controls.setFlat(True)
        self.groupBox_confocal_bottom_controls.setObjectName("groupBox_confocal_bottom_controls")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.groupBox_confocal_bottom_controls)
        self.verticalLayout_15.setContentsMargins(1, 0, 1, 0)
        self.verticalLayout_15.setSpacing(0)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.widget_4 = QtWidgets.QWidget(self.groupBox_confocal_bottom_controls)
        self.widget_4.setStyleSheet("QSlider::groove:horizontal {\n"
"                    border: 1px solid #999999;\n"
"                    height: 8px; /* the groove expands to the size of the slider by default. by giving it a height, it has a fixed size */\n"
"                    width: 100px;\n"
"                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);\n"
"                }\n"
"QSlider::handle::horizontal:disabled{\n"
"    background: #0033cc;\n"
"}\n"
"\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4,stop:1 #8f8f8f);\n"
"    border: 1px solid #5c5c5c;\n"
"    width: 18px;\n"
"    margin: -2px 0; /* handle is placed by default on the contents rect of the groove. Expand outside the groove */\n"
"    border-radius: 3px;}\n"
"QSlider::handle:horizontal:pressed {\n"
"    background: #0033cc;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: white;\n"
"                }\n"
"")
        self.widget_4.setObjectName("widget_4")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget_4)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setSpacing(0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.comboBox_channel3_color = QtWidgets.QComboBox(self.widget_4)
        self.comboBox_channel3_color.setEnabled(False)
        self.comboBox_channel3_color.setEditable(False)
        self.comboBox_channel3_color.setObjectName("comboBox_channel3_color")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icon/icons/square-rounded-32-r.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox_channel3_color.addItem(icon3, "")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icon/icons/square-rounded-32-g.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox_channel3_color.addItem(icon4, "")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icon/icons/square-rounded-32-b.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox_channel3_color.addItem(icon5, "")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icon/icons/square-rounded-32-c.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox_channel3_color.addItem(icon6, "")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/icon/icons/square-rounded-32-m.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox_channel3_color.addItem(icon7, "")
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/icon/icons/square-rounded-32-y.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox_channel3_color.addItem(icon8, "")
        self.gridLayout_3.addWidget(self.comboBox_channel3_color, 1, 3, 1, 1)
        self.comboBox_channel0_color = QtWidgets.QComboBox(self.widget_4)
        self.comboBox_channel0_color.setEnabled(False)
        self.comboBox_channel0_color.setEditable(False)
        self.comboBox_channel0_color.setObjectName("comboBox_channel0_color")
        self.comboBox_channel0_color.addItem(icon3, "")
        self.comboBox_channel0_color.addItem(icon4, "")
        self.comboBox_channel0_color.addItem(icon5, "")
        self.comboBox_channel0_color.addItem(icon6, "")
        self.comboBox_channel0_color.addItem(icon7, "")
        self.comboBox_channel0_color.addItem(icon8, "")
        self.gridLayout_3.addWidget(self.comboBox_channel0_color, 1, 0, 1, 1)
        self.comboBox_channel2_color = QtWidgets.QComboBox(self.widget_4)
        self.comboBox_channel2_color.setEnabled(False)
        self.comboBox_channel2_color.setEditable(False)
        self.comboBox_channel2_color.setObjectName("comboBox_channel2_color")
        self.comboBox_channel2_color.addItem(icon3, "")
        self.comboBox_channel2_color.addItem(icon4, "")
        self.comboBox_channel2_color.addItem(icon5, "")
        self.comboBox_channel2_color.addItem(icon6, "")
        self.comboBox_channel2_color.addItem(icon7, "")
        self.comboBox_channel2_color.addItem(icon8, "")
        self.gridLayout_3.addWidget(self.comboBox_channel2_color, 1, 2, 1, 1)
        self.slider_channel2_slice = QtWidgets.QSlider(self.widget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slider_channel2_slice.sizePolicy().hasHeightForWidth())
        self.slider_channel2_slice.setSizePolicy(sizePolicy)
        self.slider_channel2_slice.setOrientation(QtCore.Qt.Horizontal)
        self.slider_channel2_slice.setObjectName("slider_channel2_slice")
        self.gridLayout_3.addWidget(self.slider_channel2_slice, 2, 2, 1, 1)
        self.checkBox_channel0 = QtWidgets.QCheckBox(self.widget_4)
        self.checkBox_channel0.setEnabled(False)
        self.checkBox_channel0.setChecked(False)
        self.checkBox_channel0.setObjectName("checkBox_channel0")
        self.gridLayout_3.addWidget(self.checkBox_channel0, 0, 0, 1, 1)
        self.checkBox_channel2 = QtWidgets.QCheckBox(self.widget_4)
        self.checkBox_channel2.setEnabled(False)
        self.checkBox_channel2.setChecked(False)
        self.checkBox_channel2.setObjectName("checkBox_channel2")
        self.gridLayout_3.addWidget(self.checkBox_channel2, 0, 2, 1, 1)
        self.comboBox_channel1_color = QtWidgets.QComboBox(self.widget_4)
        self.comboBox_channel1_color.setEnabled(False)
        self.comboBox_channel1_color.setEditable(False)
        self.comboBox_channel1_color.setObjectName("comboBox_channel1_color")
        self.comboBox_channel1_color.addItem(icon3, "")
        self.comboBox_channel1_color.addItem(icon4, "")
        self.comboBox_channel1_color.addItem(icon5, "")
        self.comboBox_channel1_color.addItem(icon6, "")
        self.comboBox_channel1_color.addItem(icon7, "")
        self.comboBox_channel1_color.addItem(icon8, "")
        self.gridLayout_3.addWidget(self.comboBox_channel1_color, 1, 1, 1, 1)
        self.label_info = QtWidgets.QLabel(self.widget_4)
        self.label_info.setLineWidth(1)
        self.label_info.setObjectName("label_info")
        self.gridLayout_3.addWidget(self.label_info, 3, 0, 1, 4)
        self.checkBox_channel1 = QtWidgets.QCheckBox(self.widget_4)
        self.checkBox_channel1.setEnabled(False)
        self.checkBox_channel1.setStyleSheet("")
        self.checkBox_channel1.setChecked(False)
        self.checkBox_channel1.setObjectName("checkBox_channel1")
        self.gridLayout_3.addWidget(self.checkBox_channel1, 0, 1, 1, 1)
        self.slider_channel3_slice = QtWidgets.QSlider(self.widget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slider_channel3_slice.sizePolicy().hasHeightForWidth())
        self.slider_channel3_slice.setSizePolicy(sizePolicy)
        self.slider_channel3_slice.setOrientation(QtCore.Qt.Horizontal)
        self.slider_channel3_slice.setObjectName("slider_channel3_slice")
        self.gridLayout_3.addWidget(self.slider_channel3_slice, 2, 3, 1, 1)
        self.checkBox_channel3 = QtWidgets.QCheckBox(self.widget_4)
        self.checkBox_channel3.setEnabled(False)
        self.checkBox_channel3.setChecked(False)
        self.checkBox_channel3.setObjectName("checkBox_channel3")
        self.gridLayout_3.addWidget(self.checkBox_channel3, 0, 3, 1, 1)
        self.slider_channel0_slice = QtWidgets.QSlider(self.widget_4)
        self.slider_channel0_slice.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slider_channel0_slice.sizePolicy().hasHeightForWidth())
        self.slider_channel0_slice.setSizePolicy(sizePolicy)
        self.slider_channel0_slice.setStyleSheet("")
        self.slider_channel0_slice.setSingleStep(1)
        self.slider_channel0_slice.setSliderPosition(0)
        self.slider_channel0_slice.setOrientation(QtCore.Qt.Horizontal)
        self.slider_channel0_slice.setInvertedAppearance(False)
        self.slider_channel0_slice.setInvertedControls(True)
        self.slider_channel0_slice.setObjectName("slider_channel0_slice")
        self.gridLayout_3.addWidget(self.slider_channel0_slice, 2, 0, 1, 1)
        self.slider_channel1_slice = QtWidgets.QSlider(self.widget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slider_channel1_slice.sizePolicy().hasHeightForWidth())
        self.slider_channel1_slice.setSizePolicy(sizePolicy)
        self.slider_channel1_slice.setOrientation(QtCore.Qt.Horizontal)
        self.slider_channel1_slice.setObjectName("slider_channel1_slice")
        self.gridLayout_3.addWidget(self.slider_channel1_slice, 2, 1, 1, 1)
        self.verticalLayout_15.addWidget(self.widget_4)
        self.horizontalLayout_8.addWidget(self.groupBox_confocal_bottom_controls)
        self.verticalLayout_11.addWidget(self.widget_6)
        self.pushButton_process = QtWidgets.QPushButton(self.dockWidgetContents)
        self.pushButton_process.setEnabled(True)
        self.pushButton_process.setObjectName("pushButton_process")
        self.verticalLayout_11.addWidget(self.pushButton_process)
        self.progressBar = QtWidgets.QProgressBar(self.dockWidgetContents)
        self.progressBar.setStyleSheet("")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_11.addWidget(self.progressBar)
        self.dockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.dockWidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        self.tab_5.setCurrentIndex(1)
        self.comboBox_channel3_color.setCurrentIndex(3)
        self.comboBox_channel0_color.setCurrentIndex(0)
        self.comboBox_channel2_color.setCurrentIndex(2)
        self.comboBox_channel1_color.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "LineProfiler_1.0"))
        self.groupBox_top_controls.setTitle(_translate("MainWindow", "SIM images"))
        self.pushButton_open.setText(_translate("MainWindow", "Open files..."))
        self.pushButton_close_all.setText(_translate("MainWindow", "Close files"))
        self.pushButton_show.setText(_translate("MainWindow", "Show selected"))
        self.pushButton_unload.setText(_translate("MainWindow", "Unload selected"))
        self.pushButton_close.setText(_translate("MainWindow", "Close selected"))
        self.tab_5.setTabText(self.tab_5.indexOf(self.tab_3), _translate("MainWindow", "Select"))
        self.groupBox_config.setTitle(_translate("MainWindow", "Specific configurations"))
        self.label_distance_threshold.setText(_translate("MainWindow", "Operation Mode"))
        self.label_upper_lim.setText(_translate("MainWindow", "Upper distance limit [nm]"))
        self.label_lower_lim.setText(_translate("MainWindow", "Lower distance limit [nm]"))
        self.label_blur.setText(_translate("MainWindow", "Gaussian blur"))
        self.px_size_label.setText(_translate("MainWindow", "Pixel size [micro meter]"))
        self.label_3.setText(_translate("MainWindow", "Spline parameter"))
        self.comboBox_operation_mode.setItemText(0, _translate("MainWindow", "SNC one channel"))
        self.comboBox_operation_mode.setItemText(1, _translate("MainWindow", "Microtuboli"))
        self.comboBox_operation_mode.setItemText(2, _translate("MainWindow", "SNC"))
        self.groupBox_config_calibration.setTitle(_translate("MainWindow", "Intensity threshold"))
        self.label.setText(_translate("MainWindow", "Intensity value"))
        self.tab_5.setTabText(self.tab_5.indexOf(self.tab_4), _translate("MainWindow", "Config"))
        self.plotParameters.setTitle(_translate("MainWindow", "Plot parameters"))
        self.checkBox_gaussian.setText(_translate("MainWindow", "Gaussian"))
        self.checkBox_trigaussian.setText(_translate("MainWindow", "Trigaussian"))
        self.checkBox_multi_cylidner_projection.setText(_translate("MainWindow", "Multi_cylinder_projection"))
        self.checkBox_bigaussian.setText(_translate("MainWindow", "Bigaussian"))
        self.checkBox_cylinder_projection.setText(_translate("MainWindow", "Cylinder_projection"))
        self.label_2.setText(_translate("MainWindow", "Expansion factor"))
        self.tab_5.setTabText(self.tab_5.indexOf(self.tab), _translate("MainWindow", "Plot options"))
        self.dockWidget.setWindowTitle(_translate("MainWindow", "2D Widget"))
        self.groupBox_confocal_bottom_controls.setTitle(_translate("MainWindow", "SIM image"))
        self.comboBox_channel3_color.setItemText(0, _translate("MainWindow", "red"))
        self.comboBox_channel3_color.setItemText(1, _translate("MainWindow", "green"))
        self.comboBox_channel3_color.setItemText(2, _translate("MainWindow", "blue"))
        self.comboBox_channel3_color.setItemText(3, _translate("MainWindow", "cyan"))
        self.comboBox_channel3_color.setItemText(4, _translate("MainWindow", "magenta"))
        self.comboBox_channel3_color.setItemText(5, _translate("MainWindow", "yellow"))
        self.comboBox_channel0_color.setItemText(0, _translate("MainWindow", "red"))
        self.comboBox_channel0_color.setItemText(1, _translate("MainWindow", "green"))
        self.comboBox_channel0_color.setItemText(2, _translate("MainWindow", "blue"))
        self.comboBox_channel0_color.setItemText(3, _translate("MainWindow", "cyan"))
        self.comboBox_channel0_color.setItemText(4, _translate("MainWindow", "magenta"))
        self.comboBox_channel0_color.setItemText(5, _translate("MainWindow", "yellow"))
        self.comboBox_channel2_color.setItemText(0, _translate("MainWindow", "red"))
        self.comboBox_channel2_color.setItemText(1, _translate("MainWindow", "green"))
        self.comboBox_channel2_color.setItemText(2, _translate("MainWindow", "blue"))
        self.comboBox_channel2_color.setItemText(3, _translate("MainWindow", "cyan"))
        self.comboBox_channel2_color.setItemText(4, _translate("MainWindow", "magenta"))
        self.comboBox_channel2_color.setItemText(5, _translate("MainWindow", "yellow"))
        self.checkBox_channel0.setText(_translate("MainWindow", "Ch1"))
        self.checkBox_channel2.setText(_translate("MainWindow", "Ch3"))
        self.comboBox_channel1_color.setItemText(0, _translate("MainWindow", "red"))
        self.comboBox_channel1_color.setItemText(1, _translate("MainWindow", "green"))
        self.comboBox_channel1_color.setItemText(2, _translate("MainWindow", "blue"))
        self.comboBox_channel1_color.setItemText(3, _translate("MainWindow", "cyan"))
        self.comboBox_channel1_color.setItemText(4, _translate("MainWindow", "magenta"))
        self.comboBox_channel1_color.setItemText(5, _translate("MainWindow", "yellow"))
        self.label_info.setText(_translate("MainWindow", "-"))
        self.checkBox_channel1.setText(_translate("MainWindow", "Ch2"))
        self.checkBox_channel3.setText(_translate("MainWindow", "Ch4"))
        self.pushButton_process.setText(_translate("MainWindow", "Run"))


