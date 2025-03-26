import logging
import os
from typing import List

import napari
import napari.plugins
import napari.utils
import napari.utils.notifications
import numpy as np
from napari.qt.threading import thread_worker
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal

from tdct.generate import acquire_fib_view_screenshots
from tdct.io import load_and_parse_fm_image, write_ome_tiff
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
        self.filename: str = None
        self.md: dict = None
        self.setWindowTitle("Import Fluorescence Image")

        self.setup_connections()

    def setup_connections(self):
        self.pushButton_imaging_view.clicked.connect(self.on_imaging_view)
        self.pushButton_milling_view.clicked.connect(self.on_milling_view)
        self.pushButton_confirm_view.clicked.connect(self.confirm_fib_view)

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
        self.accepted.connect(self.on_accepted)
        self.rejected.connect(self.on_cancel)

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
        if pixelsize is None:
            pixelsize = 154e-9
        if zstep is None:
            zstep = 500e-9

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

        # TODO: add support for exporting images with correct metadata

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                parent=self,
                caption="Export Image as OME-TIFF",
                directory=os.path.dirname(self.filename),
                filter="OME-TIFF (*.ome.tiff);;All Files (*)",
            )

        # no filename selected
        if not filename:
            return

        # update metadata
        self.md["pixel_size"] = self.pixelsize
        self.md["zstep"] = self.zstep
        self.md["colours"] = self.colours
        self.md["is_fib_view"] = self.is_fib_view

        # write the image to file
        output_filename = write_ome_tiff(image=self.image, 
                                         md=self.md, 
                                         filename=filename)

        self.filename = output_filename
        self.label_import_header.setText(f"File: {os.path.basename(output_filename)}")
        napari.utils.notifications.show_info(f"Image exported successfully to {os.path.basename(output_filename)}")

    def on_imaging_view(self):
        self.viewer.dims.ndisplay = 3
        self.viewer.camera.angles = (0, 0, 90)

    def on_milling_view(self):
        self.viewer.dims.ndisplay = 3
        rotation = self.doubleSpinBox_rotation_x.value()                    # deg
        milling_angle = self.doubleSpinBox_milling_angle.value()            # deg
        self.viewer.camera.angles = (np.cos(np.radians(rotation)), 0, 90+52+35-milling_angle)
        # TODO: display the stage-tilt, based off pre-tilt calculation

    def confirm_fib_view(self):
        """Confirm the fib-view and overwrite the current image"""
        milling_angle = self.doubleSpinBox_milling_angle.value()            # deg

        msg = QMessageBox(parent=self)
        msg.setWindowTitle("Confirm FIB-View")
        msg.setText(f"Are you sure you want to use the current view ({milling_angle:.1f}°) ? This will overwrite the current image.")
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.exec_()

        response = (
            True
            if (msg.clickedButton() == msg.button(QMessageBox.Yes))
            or (msg.clickedButton() == msg.button(QMessageBox.Ok))
            else False
        )

        if not response:
            return
        # overwrite the exising image with the fib-view
        # set the new pixel size
        self.is_fib_view = True
        arrs = acquire_fib_view_screenshots(self.viewer)
        self.load_image(image=arrs,
                        pixelsize=self.pixelsize,
                        zstep=self.zstep,
                        colours=self.colours)

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

    def on_accepted(self):

        # TODO: emit the final data, containing the image and metadata for main app

        # TODO: get the colours from the actual layers?
        self.finished_signal.emit({"image": self.image,
                                   "pixel_size": self.pixelsize,
                                   "zstep": self.zstep,
                                   "colours" : self.colours,
                                   "is_fib_view": self.is_fib_view,
                                   "filename": self.filename})
        # self.viewer.close()

    def on_id_changed(self, page_id: int):
        logging.debug(f"id changed: {page_id}")

    def on_cancel(self):
        logging.debug("cancel pressed")
        self.viewer.close()

    def open_image(self, filename: str):
        image, md = load_and_parse_fm_image(filename)
        self.md = md # initial metadata

        self.filename = filename
        self.label_import_header.setText(f"File: {os.path.basename(filename)}")

        self.load_image(image=image,
                      pixelsize=md.get("pixel_size", 0.0),
                      zstep=md.get("zstep", 0.0),
                      colours=md.get("colours", None))

# TODO: add napari reader func??? -> requires plugin engine

def open_import_wizard(filename: str) -> FMImportWizard:
    viewer = napari.Viewer(title="FM Import Wizard")
    wizard = FMImportWizard(viewer=viewer)
    wizard.open_image(filename)
    viewer.window.add_dock_widget(wizard, name="Import Wizard")
    return wizard

def main():

    viewer = napari.Viewer()
    wizard = FMImportWizard(viewer=viewer)
    PATH = "/home/patrick/github/3DCT/3D_correlation_test_dataset/test-image2.ome.tiff"

    wizard.open_image(PATH)
    viewer.window.add_dock_widget(wizard)

    napari.run()

if __name__ == "__main__":
    main()