# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tdct_main.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(793, 953)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, -106, 747, 974))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.groupBox = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.lineEdit_project_path = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_project_path.setObjectName("lineEdit_project_path")
        self.gridLayout_9.addWidget(self.lineEdit_project_path, 0, 1, 1, 1)
        self.label_fib_image_path = QtWidgets.QLabel(self.groupBox)
        self.label_fib_image_path.setObjectName("label_fib_image_path")
        self.gridLayout_9.addWidget(self.label_fib_image_path, 1, 0, 1, 1)
        self.label_project_path = QtWidgets.QLabel(self.groupBox)
        self.label_project_path.setObjectName("label_project_path")
        self.gridLayout_9.addWidget(self.label_project_path, 0, 0, 1, 1)
        self.label_fm_image_path = QtWidgets.QLabel(self.groupBox)
        self.label_fm_image_path.setObjectName("label_fm_image_path")
        self.gridLayout_9.addWidget(self.label_fm_image_path, 2, 0, 1, 1)
        self.toolButton_project_path = QtWidgets.QToolButton(self.groupBox)
        self.toolButton_project_path.setObjectName("toolButton_project_path")
        self.gridLayout_9.addWidget(self.toolButton_project_path, 0, 2, 1, 1)
        self.toolButton_fib_image_path = QtWidgets.QToolButton(self.groupBox)
        self.toolButton_fib_image_path.setObjectName("toolButton_fib_image_path")
        self.gridLayout_9.addWidget(self.toolButton_fib_image_path, 1, 2, 1, 1)
        self.toolButton_fm_image_path = QtWidgets.QToolButton(self.groupBox)
        self.toolButton_fm_image_path.setObjectName("toolButton_fm_image_path")
        self.gridLayout_9.addWidget(self.toolButton_fm_image_path, 2, 2, 1, 1)
        self.lineEdit_fib_image_path = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_fib_image_path.setObjectName("lineEdit_fib_image_path")
        self.gridLayout_9.addWidget(self.lineEdit_fib_image_path, 1, 1, 1, 1)
        self.lineEdit_fm_image_path = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_fm_image_path.setObjectName("lineEdit_fm_image_path")
        self.gridLayout_9.addWidget(self.lineEdit_fm_image_path, 2, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 0, 0, 1, 1)
        self.groupBox_coordinates = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_coordinates.setObjectName("groupBox_coordinates")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.groupBox_coordinates)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_fib_poi_header = QtWidgets.QLabel(self.groupBox_coordinates)
        self.label_fib_poi_header.setObjectName("label_fib_poi_header")
        self.gridLayout_8.addWidget(self.label_fib_poi_header, 2, 0, 1, 1)
        self.tableWidget_fib_fid_coordinates = QtWidgets.QTableWidget(self.groupBox_coordinates)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.tableWidget_fib_fid_coordinates.setFont(font)
        self.tableWidget_fib_fid_coordinates.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        self.tableWidget_fib_fid_coordinates.setDefaultDropAction(QtCore.Qt.IgnoreAction)
        self.tableWidget_fib_fid_coordinates.setColumnCount(3)
        self.tableWidget_fib_fid_coordinates.setObjectName("tableWidget_fib_fid_coordinates")
        self.tableWidget_fib_fid_coordinates.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_fib_fid_coordinates.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_fib_fid_coordinates.setHorizontalHeaderItem(1, item)
        self.gridLayout_8.addWidget(self.tableWidget_fib_fid_coordinates, 1, 0, 1, 1)
        self.label_fib_fid_header = QtWidgets.QLabel(self.groupBox_coordinates)
        self.label_fib_fid_header.setObjectName("label_fib_fid_header")
        self.gridLayout_8.addWidget(self.label_fib_fid_header, 0, 0, 1, 1)
        self.label_fm_fid_header = QtWidgets.QLabel(self.groupBox_coordinates)
        self.label_fm_fid_header.setObjectName("label_fm_fid_header")
        self.gridLayout_8.addWidget(self.label_fm_fid_header, 0, 1, 1, 1)
        self.tableWidget_fm_fid_coordinates = QtWidgets.QTableWidget(self.groupBox_coordinates)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.tableWidget_fm_fid_coordinates.setFont(font)
        self.tableWidget_fm_fid_coordinates.setColumnCount(3)
        self.tableWidget_fm_fid_coordinates.setObjectName("tableWidget_fm_fid_coordinates")
        self.tableWidget_fm_fid_coordinates.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_fm_fid_coordinates.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_fm_fid_coordinates.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_fm_fid_coordinates.setHorizontalHeaderItem(2, item)
        self.gridLayout_8.addWidget(self.tableWidget_fm_fid_coordinates, 1, 1, 1, 1)
        self.tableWidget_fib_poi_coordinates = QtWidgets.QTableWidget(self.groupBox_coordinates)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.tableWidget_fib_poi_coordinates.setFont(font)
        self.tableWidget_fib_poi_coordinates.setObjectName("tableWidget_fib_poi_coordinates")
        self.tableWidget_fib_poi_coordinates.setColumnCount(0)
        self.tableWidget_fib_poi_coordinates.setRowCount(0)
        self.gridLayout_8.addWidget(self.tableWidget_fib_poi_coordinates, 3, 0, 1, 1)
        self.label_fm_poi_header = QtWidgets.QLabel(self.groupBox_coordinates)
        self.label_fm_poi_header.setObjectName("label_fm_poi_header")
        self.gridLayout_8.addWidget(self.label_fm_poi_header, 2, 1, 1, 1)
        self.tableWidget_fm_poi_coordinates = QtWidgets.QTableWidget(self.groupBox_coordinates)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.tableWidget_fm_poi_coordinates.setFont(font)
        self.tableWidget_fm_poi_coordinates.setObjectName("tableWidget_fm_poi_coordinates")
        self.tableWidget_fm_poi_coordinates.setColumnCount(0)
        self.tableWidget_fm_poi_coordinates.setRowCount(0)
        self.gridLayout_8.addWidget(self.tableWidget_fm_poi_coordinates, 3, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_coordinates, 2, 0, 1, 1)
        self.groupBox_results = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_results.setFont(font)
        self.groupBox_results.setFlat(False)
        self.groupBox_results.setCheckable(False)
        self.groupBox_results.setObjectName("groupBox_results")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_results)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_results_error_mean_absolute_value = QtWidgets.QLabel(self.groupBox_results)
        self.label_results_error_mean_absolute_value.setObjectName("label_results_error_mean_absolute_value")
        self.gridLayout_5.addWidget(self.label_results_error_mean_absolute_value, 6, 1, 1, 1)
        self.label_results_transform_rotation_custom_value = QtWidgets.QLabel(self.groupBox_results)
        self.label_results_transform_rotation_custom_value.setObjectName("label_results_transform_rotation_custom_value")
        self.gridLayout_5.addWidget(self.label_results_transform_rotation_custom_value, 4, 1, 1, 1)
        self.label_results_transform_rotation_zero_value = QtWidgets.QLabel(self.groupBox_results)
        self.label_results_transform_rotation_zero_value.setObjectName("label_results_transform_rotation_zero_value")
        self.gridLayout_5.addWidget(self.label_results_transform_rotation_zero_value, 3, 1, 1, 1)
        self.label_results_transform_rotation_center_custom = QtWidgets.QLabel(self.groupBox_results)
        self.label_results_transform_rotation_center_custom.setObjectName("label_results_transform_rotation_center_custom")
        self.gridLayout_5.addWidget(self.label_results_transform_rotation_center_custom, 4, 0, 1, 1)
        self.label_results_error_mean_absolute = QtWidgets.QLabel(self.groupBox_results)
        self.label_results_error_mean_absolute.setObjectName("label_results_error_mean_absolute")
        self.gridLayout_5.addWidget(self.label_results_error_mean_absolute, 6, 0, 1, 1)
        self.label_results_scale_value = QtWidgets.QLabel(self.groupBox_results)
        self.label_results_scale_value.setObjectName("label_results_scale_value")
        self.gridLayout_5.addWidget(self.label_results_scale_value, 2, 1, 1, 1)
        self.label_results_error_header = QtWidgets.QLabel(self.groupBox_results)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_results_error_header.setFont(font)
        self.label_results_error_header.setObjectName("label_results_error_header")
        self.gridLayout_5.addWidget(self.label_results_error_header, 5, 0, 1, 1)
        self.label_results_error_rms_value = QtWidgets.QLabel(self.groupBox_results)
        self.label_results_error_rms_value.setObjectName("label_results_error_rms_value")
        self.gridLayout_5.addWidget(self.label_results_error_rms_value, 7, 1, 1, 1)
        self.label_results_error_rms = QtWidgets.QLabel(self.groupBox_results)
        self.label_results_error_rms.setObjectName("label_results_error_rms")
        self.gridLayout_5.addWidget(self.label_results_error_rms, 7, 0, 1, 1)
        self.label_results_transform_euler_rotation = QtWidgets.QLabel(self.groupBox_results)
        self.label_results_transform_euler_rotation.setObjectName("label_results_transform_euler_rotation")
        self.gridLayout_5.addWidget(self.label_results_transform_euler_rotation, 1, 0, 1, 1)
        self.label_results_transform_rotation_center_zero = QtWidgets.QLabel(self.groupBox_results)
        self.label_results_transform_rotation_center_zero.setObjectName("label_results_transform_rotation_center_zero")
        self.gridLayout_5.addWidget(self.label_results_transform_rotation_center_zero, 3, 0, 1, 1)
        self.label_results_scale = QtWidgets.QLabel(self.groupBox_results)
        self.label_results_scale.setObjectName("label_results_scale")
        self.gridLayout_5.addWidget(self.label_results_scale, 2, 0, 1, 1)
        self.label_results_transform_euler_rotation_value = QtWidgets.QLabel(self.groupBox_results)
        self.label_results_transform_euler_rotation_value.setObjectName("label_results_transform_euler_rotation_value")
        self.gridLayout_5.addWidget(self.label_results_transform_euler_rotation_value, 1, 1, 1, 1)
        self.label_results_transform_header = QtWidgets.QLabel(self.groupBox_results)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_results_transform_header.setFont(font)
        self.label_results_transform_header.setObjectName("label_results_transform_header")
        self.gridLayout_5.addWidget(self.label_results_transform_header, 0, 0, 1, 1)
        self.tableWidget_results_error = QtWidgets.QTableWidget(self.groupBox_results)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.tableWidget_results_error.setFont(font)
        self.tableWidget_results_error.setObjectName("tableWidget_results_error")
        self.tableWidget_results_error.setColumnCount(0)
        self.tableWidget_results_error.setRowCount(0)
        self.gridLayout_5.addWidget(self.tableWidget_results_error, 8, 0, 1, 2)
        self.gridLayout_2.addWidget(self.groupBox_results, 4, 0, 1, 1)
        self.groupBox_parameters = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_parameters.setObjectName("groupBox_parameters")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_parameters)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_parameters_pixel_size = QtWidgets.QLabel(self.groupBox_parameters)
        self.label_parameters_pixel_size.setObjectName("label_parameters_pixel_size")
        self.gridLayout_6.addWidget(self.label_parameters_pixel_size, 0, 0, 1, 1)
        self.label_parameters_rotation_center = QtWidgets.QLabel(self.groupBox_parameters)
        self.label_parameters_rotation_center.setObjectName("label_parameters_rotation_center")
        self.gridLayout_6.addWidget(self.label_parameters_rotation_center, 1, 0, 1, 1)
        self.spinBox_parameters_rotation_center_x = QtWidgets.QSpinBox(self.groupBox_parameters)
        self.spinBox_parameters_rotation_center_x.setObjectName("spinBox_parameters_rotation_center_x")
        self.gridLayout_6.addWidget(self.spinBox_parameters_rotation_center_x, 1, 1, 1, 1)
        self.spinBox_parameters_rotation_center_y = QtWidgets.QSpinBox(self.groupBox_parameters)
        self.spinBox_parameters_rotation_center_y.setObjectName("spinBox_parameters_rotation_center_y")
        self.gridLayout_6.addWidget(self.spinBox_parameters_rotation_center_y, 1, 2, 1, 1)
        self.doubleSpinBox_parameters_pixel_size = QtWidgets.QDoubleSpinBox(self.groupBox_parameters)
        self.doubleSpinBox_parameters_pixel_size.setObjectName("doubleSpinBox_parameters_pixel_size")
        self.gridLayout_6.addWidget(self.doubleSpinBox_parameters_pixel_size, 0, 3, 1, 1)
        self.spinBox_parameters_rotation_center_z = QtWidgets.QSpinBox(self.groupBox_parameters)
        self.spinBox_parameters_rotation_center_z.setObjectName("spinBox_parameters_rotation_center_z")
        self.gridLayout_6.addWidget(self.spinBox_parameters_rotation_center_z, 1, 3, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_parameters, 1, 0, 1, 1)
        self.groupBox_options = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_options.setObjectName("groupBox_options")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox_options)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.checkBox_use_zgauss_opt = QtWidgets.QCheckBox(self.groupBox_options)
        self.checkBox_use_zgauss_opt.setObjectName("checkBox_use_zgauss_opt")
        self.gridLayout_10.addWidget(self.checkBox_use_zgauss_opt, 1, 0, 1, 1)
        self.checkBox_show_corresponding_points = QtWidgets.QCheckBox(self.groupBox_options)
        self.checkBox_show_corresponding_points.setObjectName("checkBox_show_corresponding_points")
        self.gridLayout_10.addWidget(self.checkBox_show_corresponding_points, 0, 1, 1, 1)
        self.checkBox_show_points_thick_dims = QtWidgets.QCheckBox(self.groupBox_options)
        self.checkBox_show_points_thick_dims.setObjectName("checkBox_show_points_thick_dims")
        self.gridLayout_10.addWidget(self.checkBox_show_points_thick_dims, 1, 1, 1, 1)
        self.comboBox_options_point_symbol = QtWidgets.QComboBox(self.groupBox_options)
        self.comboBox_options_point_symbol.setObjectName("comboBox_options_point_symbol")
        self.gridLayout_10.addWidget(self.comboBox_options_point_symbol, 2, 1, 1, 1)
        self.checkBox_use_mip = QtWidgets.QCheckBox(self.groupBox_options)
        self.checkBox_use_mip.setObjectName("checkBox_use_mip")
        self.gridLayout_10.addWidget(self.checkBox_use_mip, 0, 0, 1, 1)
        self.label_options_point_symbol = QtWidgets.QLabel(self.groupBox_options)
        self.label_options_point_symbol.setObjectName("label_options_point_symbol")
        self.gridLayout_10.addWidget(self.label_options_point_symbol, 2, 0, 1, 1)
        self.label_options_point_size = QtWidgets.QLabel(self.groupBox_options)
        self.label_options_point_size.setObjectName("label_options_point_size")
        self.gridLayout_10.addWidget(self.label_options_point_size, 3, 0, 1, 1)
        self.spinBox_options_point_size = QtWidgets.QSpinBox(self.groupBox_options)
        self.spinBox_options_point_size.setObjectName("spinBox_options_point_size")
        self.gridLayout_10.addWidget(self.spinBox_options_point_size, 3, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_options, 3, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 5, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 1, 0, 1, 1)
        self.pushButton_run_correlation = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_run_correlation.setObjectName("pushButton_run_correlation")
        self.gridLayout.addWidget(self.pushButton_run_correlation, 3, 0, 1, 2)
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 3)
        self.label_instructions = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_instructions.setFont(font)
        self.label_instructions.setObjectName("label_instructions")
        self.gridLayout.addWidget(self.label_instructions, 2, 0, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 793, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_FIB_Image = QtWidgets.QAction(MainWindow)
        self.actionLoad_FIB_Image.setObjectName("actionLoad_FIB_Image")
        self.actionLoad_FM_Image = QtWidgets.QAction(MainWindow)
        self.actionLoad_FM_Image.setObjectName("actionLoad_FM_Image")
        self.actionLoad_Load_Coordinates_Old = QtWidgets.QAction(MainWindow)
        self.actionLoad_Load_Coordinates_Old.setObjectName("actionLoad_Load_Coordinates_Old")
        self.actionLoad_Load_Coordinates = QtWidgets.QAction(MainWindow)
        self.actionLoad_Load_Coordinates.setObjectName("actionLoad_Load_Coordinates")
        self.actionClear_Coordinates = QtWidgets.QAction(MainWindow)
        self.actionClear_Coordinates.setObjectName("actionClear_Coordinates")
        self.menuFile.addAction(self.actionLoad_Load_Coordinates_Old)
        self.menuFile.addAction(self.actionLoad_Load_Coordinates)
        self.menuFile.addAction(self.actionClear_Coordinates)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Project"))
        self.label_fib_image_path.setText(_translate("MainWindow", "FIB Image"))
        self.label_project_path.setText(_translate("MainWindow", "Project"))
        self.label_fm_image_path.setText(_translate("MainWindow", "FM Image"))
        self.toolButton_project_path.setText(_translate("MainWindow", "..."))
        self.toolButton_fib_image_path.setText(_translate("MainWindow", "..."))
        self.toolButton_fm_image_path.setText(_translate("MainWindow", "..."))
        self.groupBox_coordinates.setTitle(_translate("MainWindow", "Coordinates"))
        self.label_fib_poi_header.setText(_translate("MainWindow", "FIB Point of Interest Coordinates"))
        item = self.tableWidget_fib_fid_coordinates.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "X"))
        item = self.tableWidget_fib_fid_coordinates.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Y"))
        self.label_fib_fid_header.setText(_translate("MainWindow", "FIB Fiducial Coordinates"))
        self.label_fm_fid_header.setText(_translate("MainWindow", "FM Fiducial Coordinates"))
        item = self.tableWidget_fm_fid_coordinates.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "X"))
        item = self.tableWidget_fm_fid_coordinates.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Y"))
        item = self.tableWidget_fm_fid_coordinates.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Z"))
        self.label_fm_poi_header.setText(_translate("MainWindow", "FM Point of Interest Coordinates"))
        self.groupBox_results.setTitle(_translate("MainWindow", "Results"))
        self.label_results_error_mean_absolute_value.setText(_translate("MainWindow", "TextLabel"))
        self.label_results_transform_rotation_custom_value.setText(_translate("MainWindow", "TextLabel"))
        self.label_results_transform_rotation_zero_value.setText(_translate("MainWindow", "TextLabel"))
        self.label_results_transform_rotation_center_custom.setText(_translate("MainWindow", "Rotation Center @ (0, 0, 0) [Custom]"))
        self.label_results_error_mean_absolute.setText(_translate("MainWindow", "Mean dx/dy"))
        self.label_results_scale_value.setText(_translate("MainWindow", "TextLabel"))
        self.label_results_error_header.setText(_translate("MainWindow", "Errors"))
        self.label_results_error_rms_value.setText(_translate("MainWindow", "TextLabel"))
        self.label_results_error_rms.setText(_translate("MainWindow", "RMS"))
        self.label_results_transform_euler_rotation.setText(_translate("MainWindow", "Euler Rotation Angles"))
        self.label_results_transform_rotation_center_zero.setText(_translate("MainWindow", "Rotation Center @ (0, 0, 0)"))
        self.label_results_scale.setText(_translate("MainWindow", "Scale"))
        self.label_results_transform_euler_rotation_value.setText(_translate("MainWindow", "TextLabel"))
        self.label_results_transform_header.setText(_translate("MainWindow", "Transformation"))
        self.groupBox_parameters.setTitle(_translate("MainWindow", "Parameters"))
        self.label_parameters_pixel_size.setText(_translate("MainWindow", "Pixel Size (um)"))
        self.label_parameters_rotation_center.setText(_translate("MainWindow", "Rotation Center (px)"))
        self.groupBox_options.setTitle(_translate("MainWindow", "Options"))
        self.checkBox_use_zgauss_opt.setText(_translate("MainWindow", "Use Z-Gaussian Fitting"))
        self.checkBox_show_corresponding_points.setText(_translate("MainWindow", "Show Corresponding Points"))
        self.checkBox_show_points_thick_dims.setText(_translate("MainWindow", "Show Points in Thick Dimensions"))
        self.checkBox_use_mip.setText(_translate("MainWindow", "Use Maximum Intensity Projection"))
        self.label_options_point_symbol.setText(_translate("MainWindow", "Point Symbol"))
        self.label_options_point_size.setText(_translate("MainWindow", "Point Size"))
        self.pushButton_run_correlation.setText(_translate("MainWindow", "Run Correlation"))
        self.label_title.setText(_translate("MainWindow", "3D Correlation Toolbox"))
        self.label_instructions.setText(_translate("MainWindow", "Instructions"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionLoad_FIB_Image.setText(_translate("MainWindow", "Load FIB Image"))
        self.actionLoad_FM_Image.setText(_translate("MainWindow", "Load FM Image"))
        self.actionLoad_Load_Coordinates_Old.setText(_translate("MainWindow", "Load Coordinates Data (Old)"))
        self.actionLoad_Load_Coordinates.setText(_translate("MainWindow", "Load Coordinates Data (New)"))
        self.actionClear_Coordinates.setText(_translate("MainWindow", "Clear Coordinates"))
