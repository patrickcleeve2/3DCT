

import napari

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
import numpy as np

from tdct.ui.qt import tdct_fib_view_synthesis
from tdct.generate import generate_fib_view, acquire_fib_view_screenshots
from typing import List


class FibViewSynthesisWidget(tdct_fib_view_synthesis.Ui_Form, QtWidgets.QWidget):
    continue_pressed_signal = pyqtSignal(dict)

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.setupUi(self)

        self.viewer = viewer
        self.fm_layers = []

        self.setup_connections()

    def setup_connections(self):
        self.pushButton_continue.clicked.connect(self.on_continue)
        self.pushButton_imaging_view.clicked.connect(self.on_imaging_view)
        self.pushButton_milling_view.clicked.connect(self.on_milling_view)

        self.doubleSpinBox_pixelsize.valueChanged.connect(self.on_scale_changed)
        self.doubleSpinBox_zstep.valueChanged.connect(self.on_scale_changed)


        # set suffix
        self.doubleSpinBox_pixelsize.setSuffix(" nm")
        self.doubleSpinBox_zstep.setSuffix(" nm")
        self.doubleSpinBox_rotation_x.setSuffix(" °")
        self.doubleSpinBox_milling_angle.setSuffix(" °")
        # set default values
        self.doubleSpinBox_rotation_x.setValue(0)
        self.doubleSpinBox_milling_angle.setValue(18)

    def load_image(self, 
                   image: np.ndarray, 
                   pixelsize: float,
                   zstep: float, 
                   colours: List[str] = None):
        """Load the fluorescence image, set channels as separate layers, set scale"""
        self.image = image

        if colours is None:
            colours = ["gray"] * image.shape[0]
        self.colours = colours
        self.pixelsize = pixelsize
        self.zstep = zstep

        self.doubleSpinBox_pixelsize.setValue(pixelsize * 1e9)
        self.doubleSpinBox_zstep.setValue(zstep * 1e9)

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
        self.viewer.dims.ndim = 3
        self.viewer.dims.ndisplay = 3
        self.viewer.dims.axis_labels = ["z", "y", "x"]

        # zoom = self.viewer.camera.zoom
        # self.viewer.camera.zoom = zoom * 2.0 # rough estimate

    def on_scale_changed(self):
        self.pixelsize = self.doubleSpinBox_pixelsize.value() * 1e-9
        self.zstep = self.doubleSpinBox_zstep.value() * 1e-9

        for layer in self.fm_layers:
            layer.scale = (self.zstep, self.pixelsize, self.pixelsize)

    def on_continue(self):

        arrs = acquire_fib_view_screenshots(self.viewer)

        self.continue_pressed_signal.emit({"fib_view": arrs})
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

# TODO: save image with metadata

def main():
    import napari

    viewer = napari.Viewer()
    widget = FibViewSynthesisWidget(viewer)
    # viewer.window.add_dock_widget(widget, name="FIB View Synthesis", add_vertical_stretch=True)

    from tdct.io import load_and_parse_fm_image
    PATH = "/home/patrick/github/3DCT/3D_correlation_test_dataset/test-image2.ome.tiff"
    image, md = load_and_parse_fm_image(PATH)
    widget.load_image(image=image,
                      pixelsize=md["pixel_size"],
                      zstep=md["zstep"],
                      colours=md["colours"])

    wizard = QtWidgets.QWizard()
    page = QtWidgets.QWizardPage()
    page.setTitle("Fib View Synthesis")
    page.setLayout(QtWidgets.QVBoxLayout())
    page.layout().addWidget(widget)
    wizard.addPage(page)
    wizard.addPage(QtWidgets.QWizardPage())
    wizard.addPage(QtWidgets.QWizardPage())
    wizard.addPage(QtWidgets.QWizardPage())
    viewer.window.add_dock_widget(wizard)

    def on_finished():
        print("finished")
        viewer.close()
    
    def on_accepted():
        print("accepted")

    def on_id_changed(page_id):
        print(f"id changed: {page_id}")
    wizard.currentIdChanged.connect(on_id_changed)
    wizard.accepted.connect(on_accepted)
    wizard.finished.connect(on_finished)


    napari.run()

if __name__ == "__main__":
    main()