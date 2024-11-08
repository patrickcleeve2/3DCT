from PyQt5.QtWidgets import (QApplication, QDialog)
from PyQt5.QtCore import Qt
import sys
from tdct.ui.qt import tdct_fm_import_dialog

import os

from tdct.util import multi_channel_interpolation
from napari.qt.threading import thread_worker
from PyQt5.QtCore import pyqtSignal
import numpy as np

class FluorescenceImportDialog(QDialog, tdct_fm_import_dialog.Ui_Dialog):
    progress_update = pyqtSignal(dict)
    def __init__(self, path: str, parent=None):
        super().__init__()
        self.setupUi(self)
        self.parent = parent

        self.path = path

        from tdct.app import _load_and_parse_fm_image
        self.image, self.md = _load_and_parse_fm_image(path)       

        self.setWindowTitle("Import Fluorescence Image")

        pixel_size = self.md.get("pixel_size", None)
        zstep = self.md.get("zstep", None)

        if pixel_size is not None:
            self.doubleSpinBox_target_step_size.setValue(pixel_size * 1e9)
        if zstep is not None:    
            self.doubleSpinBox_current_step_size.setValue(zstep * 1e9)

        self.image_interp = None

        self.setup_connections()

    def setup_connections(self):

        self.pushButton_interpolate.clicked.connect(self.on_interpolate)
        self.label_description.setText(f"File: {os.path.basename(self.path)}")

        self.progress_update.connect(self.update_progress)


        # hide progress bar
        self.progressBar_interpolation.setVisible(False)

    def update_progress(self, ddict: dict):

        val = ddict["value"]
        max = ddict["max"]
        prog = int(val / max * 100)

        self.progressBar_interpolation.setVisible(True)
        self.progressBar_interpolation.setValue(prog)
        self.progressBar_interpolation.setFormat(f"Interpolating Channel: {val+1}/{max}")

        print(f"Progress: {prog}%")

    def on_interpolate(self):

        self.progressBar_interpolation.setVisible(True)

        image = self.image
        zstep =  self.doubleSpinBox_current_step_size.value()       # nm
        pixel_size = self.doubleSpinBox_target_step_size.value()   # nm

        self.worker = self._interpolate_worker(
            image=image,
            current_pixelsize=zstep,
            target_pixelsize=pixel_size,
            method="spline"
        )
        self.worker.finished.connect(self._workflow_finished)
        self.worker.errored.connect(self._workflow_aborted)
        self.worker.start()

    def _workflow_finished(self):
        self.progressBar_interpolation.setVisible(False)
        self.worker = None

    def _workflow_aborted(self, exc):
        self.progressBar_interpolation.setVisible(False)
        self.worker = None

    @thread_worker
    def _interpolate_worker(self, 
                            image: np.ndarray, 
                            current_pixelsize:float, 
                            target_pixelsize: float, 
                            method: str = "spline"):
        """Interpolation worker"""
        image_interp = multi_channel_interpolation(
            image=image,
                                    pixelsize_in=current_pixelsize,
                                    pixelsize_out=target_pixelsize,
                                    method=method, parent_ui=self)

        self.image_interp = image_interp


# TODO: save interpolated image
# TODO: disable interaction while interpolating

PATH = "/home/patrick/github/3DCT/3D_correlation_test_dataset/test-image.ome.tiff"

def main():
    app = QApplication(sys.argv)
    dialog = FluorescenceImportDialog(path=PATH)
    result = dialog.exec_()
    
    if result == QDialog.Accepted:
        print("OK clicked")
    else:
        print("Cancel clicked")

if __name__ == '__main__':
    main()