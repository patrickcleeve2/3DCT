import csv
import datetime
import logging
import os
from pprint import pprint

import napari
import numpy as np
import pandas as pd
import tifffile as tff
import yaml
from napari.utils import notifications
from PIL import Image
from PyQt5 import QtWidgets

from tdct import correlation
# from tdct.beadPos import getzGauss
from tdct.util import get_z_gauss
from tdct.ui.qt import tdct_main

logging.basicConfig(level=logging.INFO)


# PATH = "/home/patrick/development/data/CORRELATION/3dct/3D_correlation_test_dataset"
PATH = "/home/patrick/github/3DCT/3D_correlation_test_dataset"


def parse_coordinates(fib_coord_filename: str, fm_coord_filename: str) -> list:
    """Parse the coordinates from the old style coordinate files"""

    def parse_coordinate_file(filename: str, delimiter: str = "\t") -> list:
        coords: list = []
        with open(filename) as csv_file:
            for row in csv.reader(csv_file, delimiter=delimiter):
                coords.append([field for field in row])
        return coords

    fib_coordinates = parse_coordinate_file(fib_coord_filename)
    fm_coordinates = parse_coordinate_file(fm_coord_filename)

    fib_coordinates = np.array(fib_coordinates, dtype=np.float32)
    fm_coordinates = np.array(fm_coordinates, dtype=np.float32)

    return fib_coordinates, fm_coordinates


# convert 2D image coordinates to microscope image coordinates
def convert_poi_to_microscope_coordinates(
    poi_coordinates: np.ndarray, fib_image_shape: tuple, pixel_size_um: float
) -> list:
    # image centre
    cx = float(fib_image_shape[1] * 0.5)
    cy = float(fib_image_shape[0] * 0.5)

    poi_image_coordinates: list = []

    for i in range(poi_coordinates.shape[1]):
        px = poi_coordinates[:, i]  # (x, y, z) in pixel coordinates
        px = [float(px[0]), float(px[1])]
        px_x, px_y = (
            px[0] - cx,
            cy - px[1],
        )  # point in microscope image coordinates (px)
        pt_um = (
            px_x * pixel_size_um,
            px_y * pixel_size_um,
        )  # point in microscope image coordinates (um)
        poi_image_coordinates.append(
            {"image_px": px, "px": [px_x, px_y], "px_um": [pt_um[0], pt_um[1]]}
        )

    return poi_image_coordinates


def extract_transformation_data(transf, mod_translation, reproj_3d, delta_2d) -> dict:
    # extract eulers in degrees
    eulers = transf.extract_euler(r=transf.q, mode="x", ret="one")
    eulers = eulers * 180 / np.pi

    # RMS error
    rms_error = transf.rmsError

    # difference between points after transforming 3D points to 2D
    delta_2d_mean_abs_err = np.absolute(delta_2d).mean(axis=1)

    transformation_data = {
        "transformation": {
            "scale": float(transf.s_scalar),
            "rotation_eulers": eulers.tolist(),
            "rotation_quaternion": transf.q.tolist(),
            "translation_around_rotation_center_custom": mod_translation.tolist(),
            "translation_around_rotation_center_zero": transf.d.tolist(),
        },
        "error": {
            "reprojected_3d": reproj_3d.tolist(),
            "delta_2d": delta_2d.tolist(),
            "mean_absolute_error": delta_2d_mean_abs_err.tolist(),
            "rms_error": float(rms_error),
        },
    }

    return transformation_data


def parse_correlation_result(cor_ret: list, input_data: dict) -> dict:
    # point of interest data
    spots_2d = cor_ret[2]  # (points of interest in 2D image)
    fib_image_shape = input_data["image_properties"]["fib_image_shape"]
    pixel_size_um = input_data["image_properties"]["fib_pixel_size_um"]

    poi_image_coordinates = convert_poi_to_microscope_coordinates(
        spots_2d, fib_image_shape, pixel_size_um
    )

    # transformation data
    transf = cor_ret[0]     # transformation matrix
    reproj_3d = cor_ret[1]  # reprojected 3D points to 2D points
    delta_2d = cor_ret[3]   # difference between reprojected 3D points and 2D points (in pixels)
    mod_translation = cor_ret[5]  # translation around rotation center
    transformation_data = extract_transformation_data(transf=transf, 
                                                      mod_translation=mod_translation, 
                                                      reproj_3d=reproj_3d, 
                                                      delta_2d=delta_2d)

    correlation_data = {"input": input_data, "output": {}}
    correlation_data["output"].update(transformation_data)
    correlation_data["output"].update({"poi": poi_image_coordinates})

    return correlation_data


def save_correlation_data(data: dict, path: str) -> None:
    correlation_data_filename = os.path.join(path, "correlation_data.yaml")
    with open(correlation_data_filename, "w") as file:
        yaml.dump(data, file)

    logging.info(f"Correlation data saved to: {correlation_data_filename}")


def parse_metadata(filename: str) -> np.ndarray:
    """parse metadata from a tfs tiff file"""
    # TODO: replace this with real parser versions eventually
    md = None
    with tff.TiffFile(filename) as tif:
        for page in tif.pages:
            for tag in page.tags.values():
                if tag.name == "FEI_HELIOS" or tag.code == 34682: # TFS_MD tag
                    md = tag.value
    return md


def load_image_and_metadata(filename: str) -> tuple[np.ndarray, dict]:
    image = tff.imread(filename)
    metadata = parse_metadata(filename)
    return image, metadata


def remove_metadata_bar(img: np.ndarray) -> np.ndarray:
    """Loop through the image, and check if the row is all zeros indicating the start of the metadata bar"""

    for i, row in enumerate(img):
        if not np.any(row):
            # trim the image when the first row with all zeros is found
            break
    return img[:i]


def load_and_parse_fib_image(filename: str) -> tuple[np.ndarray, float]:
    image, md = load_image_and_metadata(filename)
    pixel_size = None
    try:
        pixel_size = md["Scan"]["PixelWidth"]
    except KeyError as e:
        logging.warning(f"Pixel size not found in metadata: {e}")
        pass

    # convert to grayscale
    if image.ndim == 3:
        image = np.asarray(Image.fromarray(image).convert("L"))

    trim_metadata: bool = False
    try:
        shape = md["Image"]["ResolutionY"], md["Image"]["ResolutionX"]
        if image.shape != shape:
            logging.info(
                f"Image shape {image.shape} does not match metadata shape {shape}, likely a metadata bar present"
            )
            trim_metadata = True
    except KeyError as e:
        logging.warning(f"Image shape not found in metadata: {e}")
        pass

    # trim the image to before the first row with all zeros
    if trim_metadata:
        try:
            # crop the image to the metadata bar
            cropped_img = image[: shape[0], : shape[1]]
            # remove the metadata bar with image processing
            trimmed_img = remove_metadata_bar(image)

            logging.info(
                f"Cropped Shape: {cropped_img.shape}, Trimmed Shape: {trimmed_img.shape}"
            )
            if cropped_img.shape != trimmed_img.shape:
                raise ValueError(
                    "Cropped image shape does not match trimmed image shape"
                )

            if image.shape != trimmed_img.shape:
                logging.info(f"Image trimmed from {image.shape} to {trimmed_img.shape}")
                image = trimmed_img
        except Exception as e:
            logging.error(f"Error trimming image: {e}")
            pass

    # from pprint import pprint
    # pprint(md)

    return image, pixel_size

if not os.path.exists(PATH):
    PATH = __file__

# TODO: add preferences.yaml file to store the default settings
# TODO: add more user preferences, e.g. colors, sizes, etc.
USER_PREFERENCES = {
    "default_path": PATH,
    "show_corresponding_points": True,
    "show_thick_dims": True,
    "use_z_gauss_optim": True,
    "use_mip": False,
}

TEXT_PROPERTIES = {
    "string": "idx",  # dataframe column
    "size": 10,
    "color": "white",
    "anchor": "upper_right",
}

COORDINATE_LAYER_PROPERTIES = {
    "text": TEXT_PROPERTIES,
    "name": "Coordinates",
    "ndim": 3,
    "size": 20,
    "projection_mode": "all",
    "symbol": "ring",
    "blending": "additive",
    "opacity": 0.9,
    "coordinates": {
        "FIB": {"color": "lime", "translation": 0},
        "FM": {"color": "cyan", "translation": None},
        "POI": {"color": "magenta", "translation": None},
    },
}

LINE_LAYER_PROPERTIES = {
    "name": "Corresponding Points",
    "shape_type": "line",
    "edge_color": "white",
    "edge_width": 2,
}

RESULTS_LAYER_PROPERTIES = {
    "name": "Results",
    "ndim": 2,
    "size": 10,
    "symbol": "disc",
    "border_color": "magenta",
    "face_color": "magenta",
    "blending": "additive",
    "opacity": 0.9,
}

ERROR_TEXT_PROPERTIES = {
    "string": "abs_err",  # dataframe column
    "size": 5,
    "color": "red",
    "anchor": "upper_right",
}
REPROJECTION_LAYER_PROPERTIES = {
    "name": "Reprojected Data",
    "ndim": 2,
    "size": 5,
    "symbol": "disc",
    "face_color": "red",
    "border_color": "red",
    "opacity": 0.5,
    "text": ERROR_TEXT_PROPERTIES,
}

DATAFRAME_PROPERTIES = {
    "columns": ["x", "y", "z", "type", "color", "idx", "translation"]
}

CORRELATION_PROPERTIES = {"min_fiducial_points": 4, "min_poi_points": 1}

INSTRUCTIONS = """
To add points:
\tShift + Click:\tAdd FIB/FM Point (4 required)
\tCtrl + Click:\tAdd POI Point (1 required)

To edit points:
\tSelect the Coordinates Layer
\tEnter Selection Mode (Press 'S')
\tClick and Drag to move points
\tPress 'Delete' to remove points
\tPress 'Z' to enter Pan/Zoom Mode
"""

# add dynamic instructions for different steps

INSTRUCTIONS2 = """

\n3D Correlation Tool
\n

\nTo add points:
\nShift + Click: Add FIB/FM Point
\nCtrl + Click: Add POI Point

To edit points:
\nSelect the Coordinates Layer
\nEnter Selection Mode (Press 'S')
\nClick and Drag to select points
\nPress 'Delete' to remove points

\nWhile the Coordinates Layer is selected:
\n Press 'S' to enter Selection Mode
\n Press 'Z' to enter Pan/Zoom Mode

To Run Correlation:
\n1. Load FIB and FM Images
\n2. Select Fiducial Coordinates (at least 4)
\n3. Select Points of Interest (at least 1)
\n4. Click 'Run Correlation'
\n The results will be displayed on the FIB Image

Data is automatically saved in the project directory.
"""


def set_table_properties(table):
    table.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.ResizeToContents
    )
    # get number of columns
    for i in range(table.columnCount()):
        table.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)

    # set min height to 200
    table.setMinimumHeight(200)


class CorrelationUI(QtWidgets.QMainWindow, tdct_main.Ui_MainWindow):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.setupUi(self)

        self.viewer = viewer

        self.path = None
        self.project_loaded: bool = False
        self.images_loaded: bool = False
        self.use_z_gauss_optim: bool = False
        self.use_mip: bool = False

        self.fib_image: np.ndarray = None
        self.fm_image: np.ndarray = None

        self.translation = None
        self.fib_image_layer = None
        self.fm_image_layer = None
        self.selected_index = []

        self.rotation_center: tuple = None

        self.df = pd.DataFrame([], columns=DATAFRAME_PROPERTIES["columns"])
        self.correlation_results: dict = None

        # add a point layer for the coordinates (FIB, FM, POI)
        self.coordinates_layer = self.viewer.add_points(
            [],
            name=COORDINATE_LAYER_PROPERTIES["name"],
            ndim=COORDINATE_LAYER_PROPERTIES["ndim"],
            size=COORDINATE_LAYER_PROPERTIES["size"],
            projection_mode=COORDINATE_LAYER_PROPERTIES["projection_mode"],
            symbol=COORDINATE_LAYER_PROPERTIES["symbol"],
            text=TEXT_PROPERTIES,
            properties=self.df,
        )
        # add a line layer for the corresponding points

        self.line_layer = None  # corresponding points
        self.results_layer = None  # points of interest results
        self.reprojection_layer = None  # correlation error data

        self.setup_connections()
        self._show_project_controls()

    def _show_results_widgets(self, enable: bool):
        self.groupBox_results.setVisible(enable)

    def setup_connections(self):
        """Configure the connections for the UI elements"""

        # image / coordinate controls
        self.checkBox_use_mip.clicked.connect(self.toggle_mip)
        self.checkBox_show_corresponding_points.clicked.connect(
            self._show_corresponding_points
        )
        self.checkBox_show_points_thick_dims.clicked.connect(self._toggle_thick_dims)

        # set the default settings
        self.checkBox_use_mip.setChecked(USER_PREFERENCES["use_mip"])
        self.checkBox_show_corresponding_points.setChecked(
            USER_PREFERENCES["show_corresponding_points"]
        )
        self.checkBox_use_zgauss_opt.setChecked(USER_PREFERENCES["use_z_gauss_optim"])
        self.checkBox_show_points_thick_dims.setChecked(
            USER_PREFERENCES["show_thick_dims"]
        )

        # set the default properties for the coordinates layer
        self.comboBox_options_point_symbol.addItems(["ring", "disc", "x"])
        self.comboBox_options_point_symbol.setCurrentText(
            COORDINATE_LAYER_PROPERTIES["symbol"]
        )
        self.spinBox_options_point_size.setValue(COORDINATE_LAYER_PROPERTIES["size"])
        self.comboBox_options_point_symbol.currentIndexChanged.connect(
            self._update_user_preferences
        )
        self.spinBox_options_point_size.valueChanged.connect(
            self._update_user_preferences
        )

        # parameter controls
        self.doubleSpinBox_parameters_pixel_size.setDecimals(4)
        self.doubleSpinBox_parameters_pixel_size.setRange(0, 100)
        self.doubleSpinBox_parameters_pixel_size.valueChanged.connect(
            self._update_parameters
        )
        self.spinBox_parameters_rotation_center_x.valueChanged.connect(
            self._update_parameters
        )
        self.spinBox_parameters_rotation_center_y.valueChanged.connect(
            self._update_parameters
        )
        self.spinBox_parameters_rotation_center_z.valueChanged.connect(
            self._update_parameters
        )

        # menu actions
        self.actionLoad_Load_Coordinates_Old.triggered.connect(self.load_coordinates)
        self.actionLoad_Load_Coordinates.triggered.connect(self.load_coordinates_v2)
        self.actionClear_Coordinates.triggered.connect(self.clear_coordinates)

        # project / image controls
        self.toolButton_project_path.clicked.connect(self._set_project_path)
        self.toolButton_fib_image_path.clicked.connect(self._set_fib_image_path)
        self.toolButton_fm_image_path.clicked.connect(self._set_fm_image_path)

        # single click callbacks
        self.coordinates_layer.mouse_drag_callbacks.append(
            self.update_correlation_points
        )
        self.coordinates_layer.events.data.connect(self.coordinates_updated_from_ui)

        # correlation controls
        self.pushButton_run_correlation.clicked.connect(self.run_correlation)
        self.pushButton_run_correlation.setStyleSheet("background-color: green")
        self.label_instructions.setText(INSTRUCTIONS)

    def _show_project_controls(self):
        self.images_loaded = self.fib_image is not None and self.fm_image is not None

        # enable the controls for loading the images
        self.toolButton_fib_image_path.setEnabled(self.project_loaded)
        self.toolButton_fm_image_path.setEnabled(self.project_loaded)
        self.lineEdit_fib_image_path.setEnabled(self.project_loaded)
        self.lineEdit_fm_image_path.setEnabled(self.project_loaded)

        # toggle the image controls based on whether the project is loaded
        self.checkBox_use_mip.setEnabled(self.images_loaded)
        self.checkBox_show_corresponding_points.setEnabled(self.images_loaded)
        self.checkBox_show_points_thick_dims.setEnabled(self.images_loaded)
        self.checkBox_use_zgauss_opt.setEnabled(self.images_loaded)

        self.groupBox_options.setVisible(self.images_loaded)
        self.groupBox_parameters.setVisible(self.images_loaded)
        self.groupBox_coordinates.setVisible(self.images_loaded)

        # enable controls for loading coordinates
        self.actionLoad_Load_Coordinates_Old.setEnabled(self.images_loaded)
        self.actionLoad_Load_Coordinates.setEnabled(self.images_loaded)
        self.actionClear_Coordinates.setEnabled(self.images_loaded)

        # enable the run correlation button (requires: images loaded, coordinates loaded)
        nfib = len(self.df[self.df["type"] == "FIB"])
        nfm = len(self.df[self.df["type"] == "FM"])
        nfm_poi = len(self.df[self.df["type"] == "POI"])

        # check if there are equal number of fiducial points in both images
        fiducial_points_equal = nfib == nfm
        min_fib_fiducial_points = nfm >= CORRELATION_PROPERTIES["min_fiducial_points"]
        min_fm_fiducial_points = nfib >= CORRELATION_PROPERTIES["min_fiducial_points"]
        min_poi_points = nfm_poi >= CORRELATION_PROPERTIES["min_poi_points"]

        correlation_enabled = bool(
            self.project_loaded
            and self.images_loaded
            and fiducial_points_equal
            and min_fib_fiducial_points
            and min_fm_fiducial_points
            and min_poi_points
        )

        self.pushButton_run_correlation.setEnabled(correlation_enabled)

        # display results
        results_enabled = correlation_enabled and not self.df.empty
        results_enabled = results_enabled and self.results_layer is not None
        self._show_results_widgets(results_enabled)

    def _set_project_path(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            caption="Select Project Directory",
            directory=USER_PREFERENCES["default_path"],
        )
        if not path:
            return

        self.path = path
        self.lineEdit_project_path.setText(self.path)
        self.project_loaded = True

        # show the image controls
        self._show_project_controls()

    def _add_image_layer(self, image: np.ndarray, layer, name: str):
        """Add an image layer to the viewer, and attach a callback to update the correlation points"""
        # TODO: fix the note below
        # NOTE: we can't just update the data, because I can't figure out how to update the dtype on a napari layer properly.
        # so we have to remove the layer and re-add it for the data to be updated properly
        if layer is not None:
            self.viewer.layers.remove(layer)
            layer = None
        layer = self.viewer.add_image(
            image, name=name, blending="additive"
        )  # TODO: migrate to properties
        # add callback to update the correlation points
        layer.mouse_drag_callbacks.append(self.update_correlation_points)

        return layer

    def _set_fib_image_path(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Open FIB Image",
            directory=self.path,
            filter="Tiff Files (*.tif)",
        )
        if not filename:
            return

        # load the fib image, set the data, and set the translation
        try:
            self.fib_image, self.fib_pixel_size = load_and_parse_fib_image(filename)
        except Exception as e:
            logging.error(f"Error loading FIB Image: {e}")
            return

        if self.fib_pixel_size is not None:
            self.doubleSpinBox_parameters_pixel_size.setValue(self.fib_pixel_size * 1e6)
        else:
            logging.warning("Pixel size not found in metadata")
            notifications.show_warning(
                "Pixel size not found in metadata, please set manually"
            )

        # self.fib_image_layer.data = self.fib_image
        self.fib_image_layer = self._add_image_layer(
            image=self.fib_image, layer=self.fib_image_layer, name="FIB Image"
        )
        self.lineEdit_fib_image_path.setText(filename)

        # update the translation of the FM and POI points
        self.translation = self.fib_image.shape[1]
        COORDINATE_LAYER_PROPERTIES["coordinates"]["FM"]["translation"] = (
            self.translation
        )
        COORDINATE_LAYER_PROPERTIES["coordinates"]["POI"]["translation"] = (
            self.translation
        )

        # apply translation to fm image
        if self.fm_image is not None:
            self.fm_image_layer.translate = [0, 0, self.translation]

        self._show_project_controls()

    def _set_fm_image_path(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Open FM Image",
            directory=self.path,
            filter="Tiff Files (*.tif)",
        )
        if not filename:
            return

        # load the fm image, set the data
        try:
            self.fm_image = tff.imread(filename)
        except Exception as e:
            logging.error(f"Error loading FM Image: {e}")
            return

        # rotation center is the centre of the image volume
        halfmax_dim = int(max(self.fm_image.shape) * 0.5)
        self.rotation_center = (halfmax_dim, halfmax_dim, halfmax_dim)

        # set range of the rotation center
        self.spinBox_parameters_rotation_center_z.setRange(0, halfmax_dim * 2)
        self.spinBox_parameters_rotation_center_x.setRange(0, halfmax_dim * 2)
        self.spinBox_parameters_rotation_center_y.setRange(0, halfmax_dim * 2)
        self.spinBox_parameters_rotation_center_z.setValue(halfmax_dim)
        self.spinBox_parameters_rotation_center_x.setValue(halfmax_dim)
        self.spinBox_parameters_rotation_center_y.setValue(halfmax_dim)

        self.fm_image_layer = self._add_image_layer(
            self.fm_image, self.fm_image_layer, "FM Image"
        )
        self.lineEdit_fm_image_path.setText(filename)

        # apply translation to fm image
        if self.translation is not None:
            self.fm_image_layer.translate = [0, 0, self.translation]

        self._show_project_controls()  # TODO: change this to a callback on the data layer?

    def _update_parameters(self):
        """Update the parameters for the correlation"""
        self.fib_pixel_size = self.doubleSpinBox_parameters_pixel_size.value() * 1e-6

        self.rotation_center = (
            self.spinBox_parameters_rotation_center_x.value(),
            self.spinBox_parameters_rotation_center_y.value(),
            self.spinBox_parameters_rotation_center_z.value(),
        )

        logging.info(
            f"Parameters updated: pixel_size: {self.fib_pixel_size}, Rotation Center: {self.rotation_center}"
        )

    def _update_user_preferences(self):
        # update the user preferences
        USER_PREFERENCES["use_mip"] = self.checkBox_use_mip.isChecked()
        USER_PREFERENCES["show_corresponding_points"] = (
            self.checkBox_show_corresponding_points.isChecked()
        )
        USER_PREFERENCES["show_thick_dims"] = (
            self.checkBox_show_points_thick_dims.isChecked()
        )
        USER_PREFERENCES["use_z_gauss_optim"] = self.checkBox_use_zgauss_opt.isChecked()

        # update the coordinate layer properties
        COORDINATE_LAYER_PROPERTIES["symbol"] = (
            self.comboBox_options_point_symbol.currentText()
        )
        COORDINATE_LAYER_PROPERTIES["size"] = self.spinBox_options_point_size.value()

        if self.coordinates_layer is not None:
            length = len(self.coordinates_layer.data)
            self.coordinates_layer.symbol = [
                COORDINATE_LAYER_PROPERTIES["symbol"]
            ] * length
            self.coordinates_layer.size = [COORDINATE_LAYER_PROPERTIES["size"]] * length

    def _toggle_thick_dims(self):
        """Toggle the thickness of the points layer"""

        show_thick_dims = self.checkBox_show_points_thick_dims.isChecked()

        z_dim = 2
        if show_thick_dims:
            z_dim = self.fm_image.shape[0] * 2 if self.fm_image is not None else 500
        self.viewer.dims.thickness = (z_dim, 1, 1)

    def toggle_mip(self):
        """Toggle the MIP of the fm image"""
        self.use_mip = not self.use_mip

        # toggle mip
        if self.use_mip:
            self.fm_image_layer.data = np.amax(self.fm_image, axis=0)
        else:
            self.fm_image_layer.data = self.fm_image

    def load_coordinates(self):
        fib_coord_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Open FIB Coordinates File",
            directory=self.path,
            filter="Text Files (*.txt)",
        )

        if not fib_coord_filename:
            logging.debug("No fib coordinates `file selected")
            return

        fm_coord_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Open FM Coordinates File",
            directory=self.path,
            filter="Text Files (*.txt)",
        )

        if not fm_coord_filename:
            logging.debug("No fm coordinates file selected")
            return

        # load the coordinates from the old style coordinate files
        fib_coordinates, fm_coordinates = parse_coordinates(
            fib_coord_filename=fib_coord_filename, fm_coord_filename=fm_coord_filename
        )

        # extract the poi coordinates from the fm coordinates (old style)
        nfib = len(fib_coordinates)
        poi_coordinates = fm_coordinates[nfib:]
        fm_coordinates = fm_coordinates[:nfib]

        # add the coordinates to the dataframe
        [self.add_point("FIB", coord[::-1], True) for coord in fib_coordinates]
        [self.add_point("FM", coord[::-1], True) for coord in fm_coordinates]
        [self.add_point("POI", coord[::-1], True) for coord in poi_coordinates]

        self._dataframe_updated()

    def load_coordinates_v2(self):
        # load data.csv
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Open Coordinates File",
            directory=self.path,
            filter="CSV Files (*.csv)",
        )
        if not filename:
            logging.debug("No file selected")
            return

        # load the data, refresh the ui
        self.df = pd.read_csv(filename)
        self._dataframe_updated()

    def _display_coordinates_in_table(self):
        # clear the table
        self.tableWidget_fib_fid_coordinates.clear()
        self.tableWidget_fm_fid_coordinates.clear()

        # set the number of rows
        nfib = len(self.df[self.df["type"] == "FIB"])
        nfm = len(self.df[self.df["type"] == "FM"])
        nfm_poi = len(self.df[self.df["type"] == "POI"])

        self.tableWidget_fib_fid_coordinates.setRowCount(nfib)
        self.tableWidget_fib_fid_coordinates.setColumnCount(3)

        self.tableWidget_fm_fid_coordinates.setRowCount(nfm)
        self.tableWidget_fm_fid_coordinates.setColumnCount(3)

        self.tableWidget_fm_poi_coordinates.setRowCount(nfm_poi)
        self.tableWidget_fm_poi_coordinates.setColumnCount(3)

        # set column headers
        self.tableWidget_fib_fid_coordinates.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.tableWidget_fm_fid_coordinates.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.tableWidget_fm_poi_coordinates.setHorizontalHeaderLabels(["X", "Y", "Z"])

        set_table_properties(self.tableWidget_fib_fid_coordinates)
        set_table_properties(self.tableWidget_fm_fid_coordinates)
        set_table_properties(self.tableWidget_fm_poi_coordinates)

        def _format_data(data: list):
            return [f"{i:.2f}" for i in data]

        # add the coordinates to the table
        for i, coord in enumerate(self.df[self.df["type"] == "FIB"].values):
            x, y, z = _format_data(coord[:3])

            self.tableWidget_fib_fid_coordinates.setItem(
                i, 0, QtWidgets.QTableWidgetItem(x)
            )
            self.tableWidget_fib_fid_coordinates.setItem(
                i, 1, QtWidgets.QTableWidgetItem(y)
            )
            self.tableWidget_fib_fid_coordinates.setItem(
                i, 2, QtWidgets.QTableWidgetItem(z)
            )

        for i, coord in enumerate(self.df[self.df["type"] == "FM"].values):
            x, y, z = _format_data(coord[:3])

            self.tableWidget_fm_fid_coordinates.setItem(
                i, 0, QtWidgets.QTableWidgetItem(x)
            )
            self.tableWidget_fm_fid_coordinates.setItem(
                i, 1, QtWidgets.QTableWidgetItem(y)
            )
            self.tableWidget_fm_fid_coordinates.setItem(
                i, 2, QtWidgets.QTableWidgetItem(z)
            )

        for i, coord in enumerate(self.df[self.df["type"] == "POI"].values):
            x, y, z = _format_data(coord[:3])

            self.tableWidget_fm_poi_coordinates.setItem(
                i, 0, QtWidgets.QTableWidgetItem(x)
            )
            self.tableWidget_fm_poi_coordinates.setItem(
                i, 1, QtWidgets.QTableWidgetItem(y)
            )
            self.tableWidget_fm_poi_coordinates.setItem(
                i, 2, QtWidgets.QTableWidgetItem(z)
            )

    def run_correlation(self):
        """Run the correlation between FIB and FM coordiantes."""

        fib_coords = self.df[self.df["type"] == "FIB"][["x", "y", "z"]].values.astype(
            np.float32
        )
        fm_coords = self.df[self.df["type"] == "FM"][["x", "y", "z"]].values.astype(
            np.float32
        )
        poi_coords = self.df[self.df["type"] == "POI"][["x", "y", "z"]].values.astype(
            np.float32
        )

        logging.info(f"FIB Coordinates: {fib_coords}")
        logging.info(f"FM Coordinates: {fm_coords}")
        logging.info(f"POI Coordinates: {poi_coords}")

        fm_image = self.fm_image
        fib_image = self.fib_image
        fib_pixel_size = self.fib_pixel_size
        rotation_center = self.rotation_center

        # standard resolutions: (old), needs to be modernised
        # standard_res = [(442, 512), (884, 1024), (1768, 2048), (3536, 4096), (7072, 8192)]

        # find closest resolution
        # res = min(standard_res, key=lambda x: abs(x[0] - fib_shape[0]))

        # if fib_shape != res:
        # print(f"Clipped to closest standard resolution: {res}")

        # fib image shape minus metadata, fib_pixelsize (microns), fm_image_shape
        image_props = [fib_image.shape, fib_pixel_size * 1e6, fm_image.shape]
        logging.info(f"Image Properties: {image_props}")

        assert fib_image.ndim == 2, "FIB Image must be 2D"
        assert fm_image.ndim == 3, "FM Image must be 3D"
        assert fib_pixel_size is not None, "FIB Pixel Size must be set"
        assert rotation_center is not None, "Rotation Center must be set"
        assert isinstance(rotation_center, tuple), "Rotation Center must be a tuple"
        assert len(rotation_center) == 3, "Rotation Center must have 3 values"

        # run correlation
        self.correlation_results = run_correlation(
            fib_coords=fib_coords,
            fm_coords=fm_coords,
            poi_coords=poi_coords,
            image_props=image_props,
            rotation_center=rotation_center,
            path=self.path,
            fib_image_filename=os.path.basename(self.lineEdit_fib_image_path.text()),
            fm_image_filename=os.path.basename(self.lineEdit_fm_image_path.text()),
        )

        self._show_correlation_results(self.correlation_results)

    def _show_correlation_results(self, correlation_results: dict):
        # show the results on the fib image
        dat = []
        poi_image_coordinates = correlation_results["output"]["poi"]
        for i, coord in enumerate(poi_image_coordinates):
            dat.append(coord["image_px"][::-1])

            # image_px: image coordinates
            # px: coordinates in microscope image (centre at 0,0)
            # px_um: coordinates in microscope image (um) (centre at 0,0)

        # TODO: add the results to the main dataframe?
        if self.results_layer is not None:
            self.viewer.layers.remove(self.results_layer)
            self.results_layer = None
        text = {
            "text": [f"POI{i+1}" for i in range(len(dat))],
            "size": 10,
            "color": "white",
            "anchor": "upper_right",
        }
        self.results_layer = self.viewer.add_points(
            dat,
            name=RESULTS_LAYER_PROPERTIES["name"],
            ndim=RESULTS_LAYER_PROPERTIES["ndim"],
            size=RESULTS_LAYER_PROPERTIES["size"],
            symbol=RESULTS_LAYER_PROPERTIES["symbol"],
            border_color=RESULTS_LAYER_PROPERTIES["border_color"],
            face_color=RESULTS_LAYER_PROPERTIES["face_color"],
            blending=RESULTS_LAYER_PROPERTIES["blending"],
            opacity=RESULTS_LAYER_PROPERTIES["opacity"],
            text=text,
        )
        self.results_layer.mouse_drag_callbacks.append(self.update_correlation_points)

        # reset selection to coordiantes layer
        self.viewer.layers.selection.active = self.coordinates_layer

        self._show_results_widgets(True)

        # display the transformation data in the results tab
        trans_data = correlation_results["output"]["transformation"]
        error_data = correlation_results["output"]["error"]
        input_data = correlation_results["input"]

        # euler angles
        euler_angles = [f"{i:.2f}" for i in trans_data["rotation_eulers"]]
        # make euler angles a string with the degrees symbol between each angle
        euler_angles = "°, ".join(euler_angles) + "°"  # e.g. 90°, 90°, 90°

        self.label_results_transform_euler_rotation_value.setText(f"{euler_angles}")

        # translation
        rcustom = [f"{i:.1f}" for i in input_data["rotation_center_custom"]]
        trans_rzero = [
            f"{i:.1f}"
            for i in trans_data["translation_around_rotation_center_zero"][:2]
        ]
        trans_rcustom = [
            f"{i:.1f}"
            for i in trans_data["translation_around_rotation_center_custom"][:2]
        ]

        # convert the translation to a string
        rcustom = ", ".join(rcustom)
        trans_rzero = ", ".join(trans_rzero)
        trans_rcustom = ", ".join(trans_rcustom)

        self.label_results_transform_rotation_zero_value.setText(f"{trans_rzero}")
        self.label_results_transform_rotation_center_custom.setText(
            f"Rotation Center @ {rcustom}"
        )
        self.label_results_transform_rotation_custom_value.setText(f"{trans_rcustom}")

        # scale
        self.label_results_scale_value.setText(f"{trans_data['scale']:.2f}")

        # error
        mae = ", ".join([f"{i:.2f}" for i in error_data["mean_absolute_error"]])
        self.label_results_error_mean_absolute_value.setText(f"{mae}")
        self.label_results_error_rms_value.setText(f"{error_data['rms_error']:.2f}")

        # display the error data in the results tab
        delta_2d = np.array(error_data["delta_2d"]).T
        reproj_3d = np.array(error_data["reprojected_3d"]).T

        # format the table
        self.tableWidget_results_error.clear()
        self.tableWidget_results_error.setRowCount(delta_2d.shape[0])
        self.tableWidget_results_error.setColumnCount(2)

        # set column headers
        self.tableWidget_results_error.setHorizontalHeaderLabels(["dx", "dy"])

        set_table_properties(self.tableWidget_results_error)

        # loop through all points and and print (x, y)
        for i, (dx, dy) in enumerate(delta_2d):
            dx, dy = f"{dx:.2f}", f"{dy:.2f}"

            self.tableWidget_results_error.setItem(i, 0, QtWidgets.QTableWidgetItem(dx))
            self.tableWidget_results_error.setItem(i, 1, QtWidgets.QTableWidgetItem(dy))


        self._draw_error_data(reproj_3d)

    def update_correlation_points(self, layer, event):
        # event.position  # (z, y, x)

        if "Control" not in event.modifiers and "Shift" not in event.modifiers:
            return

        if self.fib_image is None or self.fm_image is None:
            logging.info("No images loaded")
            return

        target_layers = [self.fm_image_layer, self.fib_image_layer]

        def check_coordinates_inside_layer(event, target_layers: list):
            for target_layer in target_layers:

                def check_coords(event_position, target_layer):
                    coords = target_layer.world_to_data(event_position)

                    extent_min = target_layer.extent.data[0]  # (z, y, x)
                    extent_max = target_layer.extent.data[1]

                    # convert the above logs into a json msg
                    msgd = {
                        "target_layer": target_layer.name,
                        "event_position": event_position,
                        "coords": coords,
                        "extent_min": extent_min,
                        "extent_max": extent_max,
                    }
                    logging.debug(msgd)

                    for i, coord in enumerate(coords):
                        if coord < extent_min[i] or coord > extent_max[i]:
                            logging.debug(
                                f"Coordinate {coord} is out of bounds ({extent_min[i]}, {extent_max[i]})"
                            )
                            return False

                    return True

                if check_coords(event.position, target_layer):
                    logging.debug(
                        f"Coordinates are within bounds of {target_layer.name}"
                    )
                    return target_layer

            return None

        target_layer = check_coordinates_inside_layer(event, target_layers)

        if target_layer is None:
            logging.info("Coordinates are not within bounds of any layer")
            return

        position = target_layer.world_to_data(event.position)
        logging.debug(f"Target Layer: {target_layer.name}, Position: {position}")

        # if shift is pressed, add fiducial point to corresponding layer
        # if control is pressed, add poi point (fm)
        if "Shift" in event.modifiers:
            if target_layer == self.fib_image_layer:
                self.add_point("FIB", position)

            elif target_layer == self.fm_image_layer:
                self.add_point("FM", position)
        elif "Control" in event.modifiers:
            if target_layer == self.fm_image_layer:
                self.add_point("POI", position)

    def add_point(self, layer: str, position: list[float], from_file: bool = False):
        logging.info(f"Adding point to layer: {layer}, Position: {position}")

        # if the position is 2D (FIB), add a z coordinate
        if len(position) == 2:
            position = [0, position[0], position[1]]

        # _map the point to the coordinates layer
        clayer_props = COORDINATE_LAYER_PROPERTIES["coordinates"]
        color = clayer_props[layer]["color"]
        translation = clayer_props[layer]["translation"]

        # get the number of points of that type in the dataframe
        n = len(self.df[self.df["type"] == layer]) + 1

        # extract coordinate data
        z, y, x = position

        # z_gauss optimisation
        self.use_z_gauss_optim = self.checkBox_use_zgauss_opt.isChecked()
        if layer in ["FM", "POI"] and self.use_z_gauss_optim and not from_file:
            prev_z = z

            try:
                # getzGauss can fail, so we need to catch the exception
                # z = getzGauss(x=x, y=y, img=self.fm_image)  # threshVal, cutout
                zval, z, _ = get_z_gauss(image=self.fm_image, x=x, y=y)
                # TODO: multi-channel support
                logging.info(f"Using Z-Gauss optimisation: {z}, previous z: {prev_z}")
            except RuntimeError as e:
                logging.error(f"Error in z-gauss optimisation: {e}")
                # show a warning to the user
                notifications.show_warning(
                    f"Error in Z-Gauss optimisation, using previous z value: {prev_z}"
                )
                z = prev_z

        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    [
                        {
                            "x": x,
                            "y": y,
                            "z": z,
                            "type": layer,
                            "color": color,
                            "idx": n,
                            "translation": translation,
                        }
                    ]
                ),
            ],
            axis=0,
            ignore_index=True,
        )

        self._dataframe_updated()

    def remove_point(self, layer: str, index: int):
        pass

    def clear_coordinates(
        self,
    ):
        # remove all entries from the dataframe
        self.df = pd.DataFrame([], columns=DATAFRAME_PROPERTIES["columns"])

        # remove results
        if self.results_layer is not None:
            self.viewer.layers.remove(self.results_layer)
            self.results_layer = None

        self._dataframe_updated()

    def _dataframe_updated(self):
        """"""
        self._toggle_thick_dims()
        self._draw_points_to_layer()
        self._display_coordinates_in_table()
        self._save_dataframe()
        self._show_corresponding_points()
        self._show_project_controls()

    def _show_corresponding_points(self):
        """Show the corresponding points between the two images."""

        # remove the line layer if it exists
        if self.line_layer is not None:
            self.viewer.layers.remove(self.line_layer)
            self.line_layer = None

        if not self.checkBox_show_corresponding_points.isChecked():
            return

        if self.df.empty:
            logging.info("No data to show corresponding points")
            return

        logging.debug("Showing Corresponding Points")

        # show the matching coordinates in the FIB - FM
        fib_coords = self.df[self.df["type"] == "FIB"][["x", "y"]].values
        fm_coords = self.df[self.df["type"] == "FM"][["x", "y", "translation"]].values

        # add the translation to the x coordinates (to account for the translation of the fm image)
        fm_coords[:, 0] += fm_coords[:, 2]
        # drop the translation column
        fm_coords = fm_coords[:, :2]

        line_data = []
        for i, (fib, fm) in enumerate(zip(fib_coords, fm_coords)):
            # reverse the coordinates to match napari
            fib, fm = fib[::-1], fm[::-1]
            # draw a line between the points
            line_data.append([fib, fm])

        # self.line_layer.data = line_data  # TODO: investigate why this doesn't work
        self.line_layer = self.viewer.add_shapes(
            line_data,
            ndim=2,
            name=LINE_LAYER_PROPERTIES["name"],
            shape_type=LINE_LAYER_PROPERTIES["shape_type"],
            edge_color=LINE_LAYER_PROPERTIES["edge_color"],
            edge_width=LINE_LAYER_PROPERTIES["edge_width"],
        )
        self.line_layer.mouse_drag_callbacks.append(self.update_correlation_points)

        # reset selection to coordiantes layer
        self.viewer.layers.selection.active = self.coordinates_layer

    def _save_dataframe(self):
        # save the dataframe to a csv file
        self.df.to_csv(os.path.join(self.path, "data.csv"), index=False)

    def _draw_points_to_layer(self):
        """draw the coordinates to the points layer"""

        # reassign the data to the layer
        pt_data = self.df[["x", "y", "z", "translation"]].values

        # add the translation to the x coordinates (to account for the translation of the fm image)
        pt_data[:, 0] += pt_data[:, 3]

        # cast all the data to float32
        pt_data = pt_data.astype(np.float32)

        layer = self.coordinates_layer
        layer.events.data.disconnect(self.coordinates_updated_from_ui)
        # reverse the coordinates to match napari
        layer.data = pt_data[:, [2, 1, 0]]
        # create a color array that matches the length of the data in each frame
        layer.face_color = self.df["color"].values.astype(str)
        layer.border_color = self.df["color"].values.astype(str)
        layer.properties = self.df  # text
        layer.events.data.connect(self.coordinates_updated_from_ui)

    def _draw_error_data(self, reprojected_points: np.ndarray):
        """Draw the reprojected data to the error layer
        Args:
            reprojected_points (np.ndarray): the reprojection of the FM points

        """
        if self.reprojection_layer is not None:
            self.viewer.layers.remove(self.reprojection_layer)
            self.reprojection_layer = None

        if reprojected_points is None:
            return

        if self.df.empty:
            logging.info("No data to show error data")
            return

        # get FIB coordinates
        # fib_coords = self.df[self.df["type"] == "FIB"][["x", "y"]].values

        # add error data to fib coordinates
        # error_coords = fib_coords + error_delta
        reprojected_points = reprojected_points[:, [1, 0]]

        reprojection_text = {
            "text": [f"E{i+1}" for i in range(len(reprojected_points))],
            "size": 8,
            "color": "white",
            "anchor": "lower_left",
        }

        self.reprojection_layer = self.viewer.add_points(
            reprojected_points,
            name=REPROJECTION_LAYER_PROPERTIES["name"],
            ndim=REPROJECTION_LAYER_PROPERTIES["ndim"],
            size=REPROJECTION_LAYER_PROPERTIES["size"],
            symbol=REPROJECTION_LAYER_PROPERTIES["symbol"],
            face_color=REPROJECTION_LAYER_PROPERTIES["face_color"],
            border_color=REPROJECTION_LAYER_PROPERTIES["border_color"],
            opacity=REPROJECTION_LAYER_PROPERTIES["opacity"],
            text=reprojection_text,
        )
        self.reprojection_layer.mouse_drag_callbacks.append(self.update_correlation_points)
        self.viewer.layers.selection.active = self.coordinates_layer

    # @qthrottled(timeout=100)
    def coordinates_updated_from_ui(self, event):
        """Update the dataframe when the coordinates are updated in the UI"""

        if event.action not in ["added", "removed", "changed"]:
            return

        # get which point was moved
        index: tuple[int] = list(event.data_indices)

        if not index:
            return  # no points selected, return

        if self.df.empty:
            return

        logging.debug(
            {
                "event_type": event.type,
                "event_action": event.action,
                "event_source": event.source.name,
                "index": index,
            }
        )

        if event.action == "added":
            logging.info(f"Data Added: {index}")
            return  # don't do anything when data is added, it's handled by the add_point method

        if event.action == "changed":
            # only trigger when the user changes the data

            # TODO: prevent data from being outside of extent of the image
            data = event.source.data

            # update the dataframe
            self.df.loc[index, ["x", "y", "z"]] = data[index][:, ::-1]

            # subtract the translation from the x coordinate, if it's an FM point
            trans_indices = self.df[self.df["type"].isin(["FM", "POI"])].index
            trans_changed_index = np.intersect1d(index, trans_indices)
            self.df.loc[trans_changed_index, "x"] -= self.df.loc[
                trans_changed_index, "translation"
            ]

        if event.action == "removed":
            # remove the point from the dataframe
            self.df.drop(index, inplace=True)

            # reset the index
            self.df.reset_index(inplace=True, drop=True)

            # rename the idx column to match the number of types of points
            self.df["idx"] = self.df.groupby("type").cumcount() + 1

        self._dataframe_updated()

        return


def run_correlation(
    fib_coords: np.ndarray,
    fm_coords: np.ndarray,
    poi_coords: np.ndarray,
    image_props: tuple,
    rotation_center: tuple,
    path: str,
    fib_image_filename: str = "",
    fm_image_filename: str = "",
) -> dict:
    """Run the correlation between the FIB and FM images"""

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file = os.path.join(path, f"{timestamp}_correlation.txt")

    # run the correlation
    correlation_results = correlation.main(
        markers_3d=fm_coords,
        markers_2d=fib_coords,
        spots_3d=poi_coords,
        rotation_center=rotation_center,
        results_file=results_file,
        imageProps=image_props,
    )

    # input data
    input_data = {
        "fib_coordinates": fib_coords.tolist(),
        "fm_coordinates": fm_coords.tolist(),
        "poi_coordinates": poi_coords.tolist(),
        "image_properties": {
            "fib_image_filename": fib_image_filename,
            "fib_image_shape": list(image_props[0]),
            "fib_pixel_size_um": float(image_props[1]),
            "fm_image_filename": fm_image_filename,
            "fm_image_shape": list(image_props[2]),
        },
        "rotation_center": list(rotation_center),
        "rotation_center_custom": list(rotation_center),
    }

    # output data
    correlation_data = parse_correlation_result(
        correlation_results, input_data=input_data
    )

    # full correlation data
    full_correlation_data = {
        "metadata": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "data_path": path,
            "csv_path": "",
            "project_path": "",
        },
        "correlation": correlation_data,
    }
    save_correlation_data(full_correlation_data, path)

    return correlation_data


# TODO:


# link the facecolor / bordercolor to the dataframe

# add 'project' concept, save directory, save project file
# add 'open project' concept, load project file, load images, load coordinates
# add an auto-save toggle

# functionalise correlation code

# add data tools: reslice (iterpolate), set pixel size, set origin, set rotation


# think about how to embed in autolamella with targetting

# display the transformation matrix
# display transformed image onto fib image
# integrate the find beads tool

# add dynamic instructions
# add show instructions toggle

# add a 'help' tab
# add an options tab

# create an iterative optimiser for correlation refinement

# update to use qtableview with pandas model
# https://doc.qt.io/qtforpython-6/examples/example_external_pandas.html
# https://stackoverflow.com/questions/47020995/pyqt5-updating-dataframe-behind-qtablewidget
# allow editting of the coordinates in the table
# allow re-ordering of the coordinates in the table

# DONE:
# load images
# show text on the points (point index)
# add z-gauss fitting to the points in fm
# when the correlation is run, show the results on the fib image
# add a 'clear' button to clear all the points
# show the corresponding points on the fm image
# add a toggle for showing 'thick' dimensions
# add a toggle for showing the mip of the fm image

# remove the metadata bar from fib image (and auto detect?)
# editable pixelsize for fib image + warning when not set
# editable rotation center
# add options to set the size / symbol / color of the points
# show the error data on the image

# connect results layer and corresponding points layer to click events, so user doesn't have to select the layer
# when a point is moved, update the dataframe
# when a point is deleted, update the dataframe
# when a point is added, update the dataframe
# save the data as data.csv
# when the correlation is run, save the results to a file

# results
# show results in results tab

# project
# create flow: create / load project -> load images -> select / load coordinates -> run correlation -> save results
# save the project file as a yaml file


def main():
    viewer = napari.Viewer()
    widget = CorrelationUI(viewer)

    # create a button to run correlation
    viewer.window.add_dock_widget(
        widget, area="right", name="3DCT Correlation", tabify=True
    )

    napari.run()


if __name__ == "__main__":
    main()
