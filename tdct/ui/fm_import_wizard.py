import logging
import os
from typing import List

import napari
import napari.plugins
import napari.utils
import numpy as np
from napari.qt.threading import thread_worker
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

from tdct.generate import acquire_fib_view_screenshots
from tdct.io import load_and_parse_fm_image
from tdct.ui.qt import tdct_fm_import_wizard as tdct_wizard
from tdct.util import INTERPOLATION_METHODS, multi_channel_interpolation

logging.basicConfig(level=logging.INFO)

# TODO: save interpolated image
# TODO: disable interaction while interpolating

class FMImportWizard(tdct_wizard.Ui_Wizard, QtWidgets.QWizard):
    finished_signal = pyqtSignal(dict)
    progress_update = pyqtSignal(dict)

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.setupUi(self)

        self.viewer = viewer
        self.fm_layers = []
        self.image: np.ndarray = None
        self.image_interp: np.ndarray = None
        self.colours: List[str] = None
        self.pixelsize: float = None
        self.zstep: float = None
        self.is_fib_view: bool = False
        self.setWindowTitle("Import Fluorescence Image")

        self.setup_connections()

    def setup_connections(self):
        self.pushButton_imaging_view.clicked.connect(self.on_imaging_view)
        self.pushButton_milling_view.clicked.connect(self.on_milling_view)

        self.doubleSpinBox_pixelsize_xy.valueChanged.connect(self.on_scale_changed)
        self.doubleSpinBox_zstep_size.valueChanged.connect(self.on_scale_changed)

        self.doubleSpinBox_current_zstep_size.valueChanged.connect(self.on_interpolation_changed)
        self.doubleSpinBox_target_zstep_size.valueChanged.connect(self.on_interpolation_changed)

        # set suffix
        self.doubleSpinBox_pixelsize_xy.setSuffix(" nm")
        self.doubleSpinBox_zstep_size.setSuffix(" nm")
        self.doubleSpinBox_current_zstep_size.setSuffix(" nm")
        self.doubleSpinBox_target_zstep_size.setSuffix(" nm")
        self.doubleSpinBox_rotation_x.setSuffix(" °")
        self.doubleSpinBox_milling_angle.setSuffix(" °")
        # set default values
        self.doubleSpinBox_rotation_x.setValue(0)
        self.doubleSpinBox_milling_angle.setValue(18)

        # set limits
        self.doubleSpinBox_pixelsize_xy.setRange(0, 1e6)
        self.doubleSpinBox_zstep_size.setRange(0, 1e6)
        self.doubleSpinBox_current_zstep_size.setRange(0, 1e6)
        self.doubleSpinBox_target_zstep_size.setRange(0, 1e6)
        self.doubleSpinBox_rotation_x.setRange(-360, 360)
        self.doubleSpinBox_milling_angle.setRange(-360, 360)

        self.comboBox_interpolation_method.addItems(INTERPOLATION_METHODS)
        self.comboBox_interpolation_method.setCurrentText(INTERPOLATION_METHODS[0])

        self.pushButton_interpolation.clicked.connect(self.on_interpolate)
        self.progress_update.connect(self.update_progress)

        # hide progress bar
        self.progressBar_interpolation.setVisible(False)

        self.currentIdChanged.connect(self.on_id_changed)
        self.finished.connect(self.on_finished)


        self.pushButton_export_image.clicked.connect(self.on_export)

    def load_image(self, 
                   image: np.ndarray, 
                   pixelsize: float,
                   zstep: float, 
                   colours: List[str] = None):
        """Load the fluorescence image, set channels as separate layers, set scale"""

        # clear existing layers
        if self.fm_layers:
            for layer in self.fm_layers:
                self.viewer.layers.remove(layer)
        self.fm_layers = []

        self.image = image

        if colours is None:
            colours = ["gray"] * image.shape[0]
        self.colours = colours
        self.pixelsize = pixelsize
        self.zstep = zstep

        self.doubleSpinBox_pixelsize_xy.setValue(pixelsize * 1e9)
        self.doubleSpinBox_zstep_size.setValue(zstep * 1e9)
        self.doubleSpinBox_current_zstep_size.setValue(zstep * 1e9)
        self.doubleSpinBox_target_zstep_size.setValue(pixelsize * 1e9) # assume isotropic is desired
        self.label_interpolation_information.setText(f"Pixelsize (x,y): {pixelsize*1e9:.1f} nm")

        for i in range(image.shape[0]):
            arr = image[i]
            colour = colours[i]
            layer = self.viewer.add_image(data=arr,
                         name=f"Channel {i}",
                         scale=(self.zstep, self.pixelsize, self.pixelsize),
                         blending="additive",
                         colormap=colour)
            self.fm_layers.append(layer)

        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "m"
        # self.viewer.dims.ndim = 3
        self.viewer.dims.ndisplay = 2
        self.viewer.dims.axis_labels = ["z", "y", "x"]
    
    def on_interpolation_changed(self):
        pass

    def on_scale_changed(self):
        self.pixelsize = self.doubleSpinBox_pixelsize_xy.value() * 1e-9
        self.zstep = self.doubleSpinBox_zstep_size.value() * 1e-9 # TODO: use constants

        for layer in self.fm_layers:
            layer.scale = (self.zstep, self.pixelsize, self.pixelsize)

    def on_export(self):

        # conditional, 
        # fib-view-synthesis: use acquire_fib_view_screenshots
        # else: use .data directly
        # query: initial metadata?

        arrs = acquire_fib_view_screenshots(self.viewer)

        # self.continue_pressed_signal.emit({"fib_view": arrs})
        print(f"continue pressed: 'fib_view': {arrs.shape}")

    def on_imaging_view(self):
        self.viewer.dims.ndisplay = 3
        self.viewer.camera.angles = (0, 0, 90)

    def on_milling_view(self):
        self.viewer.dims.ndisplay = 3
        rotation = self.doubleSpinBox_rotation_x.value()                    # deg
        milling_angle = self.doubleSpinBox_milling_angle.value()            # deg
        self.viewer.camera.angles = (np.cos(rotation), 0, milling_angle)
        # TODO: display the stage-tilt, based off pre-tilt calculation

    def confirm_fib_view(self):
        # overwrite the exising image with the fib-view
        # set the new pixel size
        self.is_fib_view = True
        pass

    def update_progress(self, ddict: dict):
        val = ddict["value"]
        max = ddict["max"]
        prog = int(val / max * 100)

        msg = f"Interpolating Channel: {val+1}/{max}"
        if val == max:
            msg = "Finalizing interpolation..."

        self.progressBar_interpolation.setVisible(True)
        self.progressBar_interpolation.setValue(prog)
        self.progressBar_interpolation.setFormat(msg)

    def on_interpolate(self):
        self.progressBar_interpolation.setVisible(True)

        image = self.image
        zstep = self.doubleSpinBox_current_zstep_size.value()  # nm
        target_pixelsize = self.doubleSpinBox_target_zstep_size.value()  # nm
        interpolation_method = self.comboBox_interpolation_method.currentText()

        self.worker = self._interpolate_worker(
            image=image,
            current_pixelsize=zstep,
            target_pixelsize=target_pixelsize,
            method=interpolation_method,
        )
        self.worker.finished.connect(self._workflow_finished)
        self.worker.errored.connect(self._workflow_aborted)
        self.worker.start()

    def _workflow_finished(self):
        self.progressBar_interpolation.setVisible(False)
        self.worker = None

        if self.image_interp is not None:
            self.load_image(image=self.image_interp,
                            pixelsize=self.pixelsize,
                            zstep=self.doubleSpinBox_target_zstep_size.value() * 1e-9,
                            colours=self.colours)

            # napari notification
            napari.utils.notifications.show_info("Interpolation finished")

    def _workflow_aborted(self, exc):
        self.progressBar_interpolation.setVisible(False)
        self.worker = None

    @thread_worker
    def _interpolate_worker(
        self,
        image: np.ndarray,
        current_pixelsize: float,
        target_pixelsize: float,
        method: str = "linear",
    ):
        """Interpolation worker"""
        image_interp = multi_channel_interpolation(
            image=image,
            pixelsize_in=current_pixelsize,
            pixelsize_out=target_pixelsize,
            method=method,
            parent_ui=self,
        )

        self.image_interp = image_interp

    def on_finished(self):
        print("finished")
        # TODO: emit the final data, containing the image and metadata for main app
        logging.info("FMImportWizard finished")

        self.finished_signal.emit({"image": self.image,
                                   "pixelsize": self.pixelsize,
                                   "zstep": self.zstep,
                                   "colours" : self.colours,
                                   "is_fib_view": self.is_fib_view})
        # TODO: get the colours from the actual layers?

        self.viewer.close()

    def on_id_changed(self, page_id: int):
        print(f"id changed: {page_id}")

    def open_image(self, path: str):
        image, md = load_and_parse_fm_image(path)

        self.path = path
        self.label_import_header.setText(f"File: {os.path.basename(path)}")

        self.load_image(image=image,
                      pixelsize=md.get("pixel_size", 0.0),
                      zstep=md.get("zstep", 0.0),
                      colours=md.get("colours", None))

# TODO: add napari reader func??? -> requires plugin engine

def open_import_wizard(path: str):
    viewer = napari.Viewer(title="FM Import Wizard")
    wizard = FMImportWizard(viewer=viewer)
    wizard.open_image(path)
    viewer.window.add_dock_widget(wizard, name="Import Wizard")
    napari.run(max_loop_level = 2)

def main():

    viewer = napari.Viewer()
    wizard = FMImportWizard(viewer=viewer)
    PATH = "/home/patrick/github/3DCT/3D_correlation_test_dataset/test-image2.ome.tiff"

    wizard.open_image(PATH)
    viewer.window.add_dock_widget(wizard)

    napari.run()

if __name__ == "__main__":
    main()