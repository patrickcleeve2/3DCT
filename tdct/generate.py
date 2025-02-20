import napari
import numpy as np


def generate_fib_view(image: np.ndarray, 
                      md: dict, 
                      milling_angle: float, 
                      rotation: float = 0, 
                      viewer: napari.Viewer = None) -> np.ndarray:
    """Generate the FIB view from the FM image stack
    Args:
        image (np.ndarray): 4D image stack (CZYX)
        md (dict): metadata
        milling_angle (float): milling angle
        rotation (float, optional): rotation around z-axis. Defaults to 0."""
    if viewer is None:
        viewer = napari.Viewer(title="FIB View Synthesis")
        napari.run()

    pz = md["zstep"]
    px = md["pixel_size"]

    for i in range(image.shape[0]):
        arr = image[i]
        colour = md["colours"][i]
        viewer.add_image(data=arr,
                         name=f"Channel {i}",
                         scale=(pz, px, px),
                         blending="additive",
                         colormap=colour)

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "m"
    viewer.dims.ndim = 3
    viewer.dims.ndisplay = 3
    viewer.dims.axis_labels = ["z", "y", "x"]
    viewer.camera.angles = (np.cos(rotation), 0, milling_angle)

    zoom = viewer.camera.zoom
    viewer.camera.zoom =  zoom * 2.0 # rough estimate

    return acquire_fib_view_screenshots(viewer)


def acquire_fib_view_screenshots(viewer: napari.Viewer) -> list[np.ndarray]:
    """Acquire screenshots of the FIB views
    Args:
        viewer (napari.Viewer): napari viewer
    Returns:
        list[np.ndarray]: list of fib-view array views"""
    
    layers = viewer.layers
    cmaps = [l.colormap for l in layers]
    arrs = []
    viewer.scale_bar.visible = False
    for layer in layers:
        # set all layers to invisible
        for l in layers:
            l.visible = False

        # set current layer to visible
        layer.visible = True
        layer.colormap = "gray"
        sc = viewer.screenshot()

        # convert from RGb to float32
        sc = sc.astype(np.float32) / 255
        arrs.append(sc[:, :, 0]) # 4D->2D

        # TODO: need to crop extra black space

    # restore visibility
    viewer.scale_bar.visible = True 
    for i, l in enumerate(layers):
        l.visible = True
        l.colormap = cmaps[i]

    # viewer.close()

    return arrs