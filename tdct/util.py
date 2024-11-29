import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, ndimage
from scipy.optimize import curve_fit, leastsq

# migrated from tdct.beadPos and refactored

### GAUSSIAN FITTING ###
def get_z_gauss(image: np.ndarray, x: int, y: int, show: bool = False) -> Tuple[float, int, float]:
    """Get the best fitting z-value for a 2D point in an ZYX image using a 1D Gaussian fit:
    Args:
        x: x coordinate
        y: y coordinate
        image: 3D numpy array (Z,Y,X)
        show: show the plot of the fit (for debugging)
    Returns:
        z: z coordinate (index)
    """

    # check that img is ndim=3
    if image.ndim != 3:
        raise ValueError(f"img must be a ZYX array, got {image.ndim}")

    # check that x, y are inside the image shape
    if x >= image.shape[-1] or y >= image.shape[-2]:
        raise ValueError(
            f"x and y must be within the image shape, x: {x}, y: {y}, {image.shape}"
        )

    # NOTE: round is important, don't directly cast to int
    if not isinstance(x, int):
        x = round(x)
    if not isinstance(y, int):
        y = round(y)

    # fit the z data for the given x,y
    poptZ, pcov = fit_guass1d(image[:, y, x], show=show)

    return np.array(poptZ)  # zval, zidx, zsigma


def fit_guass1d(data: np.ndarray, show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 1D Gaussian to the data
    Args:
        data: 1D numpy array
        show: show the plot of the fit (for debugging)
    Returns:
        popt: optimal parameters
        pcov: covariance matrix
    """

    data = data - data.min()  # shift data to 0
    p0 = [data.max(), data.argmax(), 1]  # initial guess
    x = np.arange(len(data))  # x values
    popt, pcov = curve_fit(gauss1d, x, data, p0=p0)

    # plot the data and the fit
    if show:
        plt.title("1D Gaussian fit")
        plt.plot(data, label="Data")
        plt.plot(gauss1d(x, *popt), label="Gaussian 1D fit")
        plt.legend()
        plt.show()

    return popt, pcov

def gauss1d(x: np.ndarray, A: float, mu: float, sigma: float) -> float:
    """Gaussian 1D fit
    Args:
        x: x values
        A: magnitude
        mu: offset on x axis
        sigma: width
    Returns:
        y: gaussian values
    """
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))

##### 2D GAUSSIAN FIT #####

## Gaussian 2D fit from http://scipy.github.io/old-wiki/pages/Cookbook/FittingData
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a Gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the Gaussian parameters of a 2D distribution by calculating its
    moments"""
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Returns (height, x, y, width_x, width_y)
    the Gaussian parameters of a 2D distribution found by a fit"""

    def errorfunction(p):
        return np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)

    try:
        params = moments(data)
    except ValueError:
        return None
    p, success = leastsq(errorfunction, params)
    if np.isnan(p).any():
        return None

    return p

def extract_image_patch(img: np.ndarray, x: int, y:int, z: int, cutout: int) -> np.ndarray:
    # Get image dimensions
    z_max, height, width = img.shape
    
    # Calculate patch bounds
    x_min = int(x - cutout)
    x_max = int(x + cutout)
    y_min = int(y - cutout)
    y_max = int(y + cutout)
    z = int(round(z))

    # Check if patch is within bounds
    is_valid = (
        x_min >= cutout and
        x_max < width and
        y_min >= cutout and
        y_max < height and
        0 <= z < z_max
    )

    if not is_valid:
        print("Point(s) too close to edge or out of bounds.")
        return None

    # Extract and return the patch
    return np.copy(img[z, y_min:y_max, x_min:x_max])

def threshold_image(data: np.ndarray, threshold_val: float):
    """
    Zero out values below a threshold relative to the data range.
    
    Args:
        data: numpy array of image data
        threshold_percent: float between 0-1, normalized threshold value
    """
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    
    # Calculate threshold value
    threshold_value = data_max - (data_range * threshold_val)
    
    # Zero out values below threshold 
    data[data < threshold_value] = 0
    
    return data

#### INTERPOLATION ####

INTERPOLATION_METHODS = ["linear", "spline", "fast-cubic"]

def interpolate_z_stack(
    image: np.ndarray, pixelsize_in: float, pixelsize_out: float, method: str = "linear"
) -> np.ndarray:
    return interpol(image, pixelsize_in, pixelsize_out, method)


def interpol(
    img: np.ndarray, ss_in: float, ss_out: float, method: str = "linear"
) -> np.ndarray:
    """Main function for interpolating image stacks via polyfit"""

    # check for multi-channel images, only single channel images are supported currently
    if img.ndim == 4:
        if img.shape[0] != 1:
            raise ValueError(
                f"Muli-channel images are not supported, got an image of shape {img.shape}, expected (1, Z, Y, X) or (Z, Y, X)"
            )

        # remove the channel dimension
        logging.info(f"Squeezing channel dimension from image of shape {img.shape}")
        img = np.squeeze(img, axis=0)

    if img.ndim != 3:
        raise ValueError(f"image must be a ZYX array, but got {img.ndim}")

    if method not in INTERPOLATION_METHODS:
        raise ValueError(
            f"interpolation method  must be in {INTERPOLATION_METHODS} got {method}"
        )

    # interpolate the image
    if method == "linear":
        return linear_interpolate(img, ss_in, ss_out)
    if method == "spline":
        return spline_interpolate(img, ss_in, ss_out)
    if method == "fast-cubic":
        return fast_cubic_interpolate(img, ss_in, ss_out)


def get_interpolated_shape(
    image: np.ndarray, ss_in: float, ss_out: float
) -> Tuple[int, int, int]:
    """Get the shape of the interpolated image stack
    Args:
        image: 3D numpy array (Z,Y,X)
        ss_in: step size of the input stack
        ss_out: step size of the output stack
    Returns:
        shape: shape of the interpolated image stack"""

    # # Discarding last data point. e.g. 56 in i.e.
    # # 55 steps * (309 nm original spacing / 161.25 nm new spacing) = 105.39 -> int() = 105 + 1 = 106
    sl_in = image.shape[0]  # Number of slices in original stack
    sl_out = (
        int((sl_in - 1) * (ss_in / ss_out)) + 1
    )  # Number of slices in interpolated stack
    return (sl_out, image.shape[1], image.shape[2])


def showgraph_(img, ss_in, ss_out, block=True):
    """Show graph for polyfit function to visualize fitting process"""

    # calculate the interpolated shape
    interpolated_shape = get_interpolated_shape(img, ss_in, ss_out)

    sl_in = img.shape[0]
    sl_out = interpolated_shape[0]

    ## Known x values in interpolated stack size.
    zx = np.arange(0, sl_out, (ss_in / ss_out))
    zy = img[:, int(img.shape[1] / 2), int(img.shape[2] / 2)]
    if ss_in / ss_out < 1.0:
        zx_mod = []
        for i in range(img.shape[0]):
            zx_mod.append(zx[i])
        zx = zx_mod
    ## Linear interpolation
    lin = interpolate.interp1d(zx, zy, kind="linear")
    ## Spline interpolation
    spl = interpolate.InterpolatedUnivariateSpline(zx, zy)

    zxnew = np.arange(
        0, (sl_in - 1) * ss_in / ss_out, 1
    )  # First slice of original and interpolated are both 0. n-1 to discard last slice
    zynew_lin = lin(zxnew)
    zynew_spl = spl(zxnew)
    ## Plotting data. blue = original, red = interpolated with interp1d, green = spline interpolation

    plt.plot(zx, zy, "bo-", label="original")
    plt.plot(zxnew, zynew_lin, "rx-", label="linear")
    plt.plot(zxnew, zynew_spl, "g*-", label="spline")
    plt.legend(loc="upper left")
    plt.show()


def spline_interpolate(img: np.ndarray, ss_in: float, ss_out: float) -> np.ndarray:
    """
    Spline interpolation (1D) of an image stack -> SLOW

    # possible depricated due to changes in code -> marked for futur code changes
    ss_in : step size input stack
    ss_out : step size output stack
    """

    # calculate the interpolated shape
    interpolated_shape = get_interpolated_shape(img, ss_in, ss_out)

    sl_in = img.shape[0]
    sl_out = interpolated_shape[0]
    ## Known x values in interpolated stack size.
    zx = np.arange(0, sl_out, ss_in / ss_out)
    zxnew = np.arange(
        0, (sl_in - 1) * ss_in / ss_out, 1
    )  # First slice of original and interpolated are both 0. n-1 to discard last slice
    if ss_in / ss_out < 1.0:
        zx_mod = []
        for i in range(img.shape[0]):
            zx_mod.append(zx[i])
        zx = zx_mod

    ## Create new numpy array for the interpolated image stack
    img_interpolated = np.zeros(shape=interpolated_shape, dtype=img.dtype)

    r_sl_out = list(range(sl_out))

    # pixelwise interpolation -> very slow, is there a 2D version of this? #TODO: speed up
    for px in range(img.shape[-1]):
        for py in range(img.shape[-2]):
            # logging.info(f"Interpolating pixel ({px}/{img.shape[-1]}, {py}/{img.shape[-2]})")
            spl = interpolate.InterpolatedUnivariateSpline(zx, img[:, py, px])
            np.put(img_interpolated[:, py, px], r_sl_out, spl(zxnew))

    return img_interpolated


def linear_interpolate(img: np.ndarray, ss_in: float, ss_out: float) -> np.ndarray:
    """Linear interpolation"""

    # calculate the interpolated shape
    interpolated_shape = get_interpolated_shape(img, ss_in, ss_out)

    sl_in = img.shape[0]
    sl_out = interpolated_shape[0]

    ##  Determine interpolated slice positions
    sl_int = np.arange(
        0, sl_in - 1, ss_out / ss_in
    )  # sl_in-1 because last slice is discarded (no extrapolation)

    ## Create new numpy array for the interpolated image stack
    img_int = np.zeros(shape=interpolated_shape, dtype=img.dtype)

    ## Calculate distances from every interpolated image to its next original image
    sl_counter = 0
    for i in sl_int:
        int_i = int(i)
        lower = i - int_i
        upper = 1 - (lower)
        img_int[sl_counter, :, :] = (
            img[int_i, :, :] * upper + img[int_i + 1, :, :] * lower
        )
        sl_counter += 1
    return img_int


def fast_cubic_interpolate(
    image_3d: np.ndarray, original_z_size: float, target_z_size: float
) -> np.ndarray:
    """
    Fast interpolation of a 3D image array along the z-axis using scipy's zoom function.

    Parameters:
    -----------
    image_3d : ndarray
        Input 3D image array with shape (Z, Y, X)
    original_z_size : float
        Original pixel size in z-axis (e.g., 10 for 10µm)
    target_z_size : float
        Desired pixel size in z-axis (e.g., 5 for 5µm)

    Returns:
    --------
    ndarray
        Interpolated 3D image with adjusted z-axis resolution
    """
    # Calculate the scaling factor
    scale_factor = original_z_size / target_z_size

    # Create zoom factors for each dimension
    # Only scale the z-axis (first dimension)
    zoom_factors = (scale_factor, 1, 1)

    # Perform the interpolation
    # order=3 for cubic interpolation
    # mode='reflect' to handle edge cases
    # prefilter=True for better quality
    interpolated = ndimage.zoom(
        image_3d, zoom_factors, order=3, mode="reflect", prefilter=True
    )

    return interpolated


#### multi-channel interpolation ####

def multi_channel_interpolation(
    image: np.ndarray,
    pixelsize_in: float,
    pixelsize_out: float,
    method: str = "fast-cubic",
    parent_ui=None,
) -> np.ndarray:
    """Interpolate a multi-channel z-stack (CZYX) along the z-axis
    Args:
        image: 4D numpy array (CZYX)
        pixelsize_in: original pixel size in z-axis
        pixelsize_out: desired pixel size in z-axis
    Returns:
        interpolated: 4D numpy array (CZYX) with adjusted z-axis resolution
    """
    if parent_ui:
        parent_ui.progress_update.emit({"value": 0, "max": image.shape[0]})

    # QUERY: how to speed up?
    ch_interpolated = []
    for i, channel in enumerate(image):
        logging.info(f"Interpolating channel {i+1}/{image.shape[0]}")
        ch_interpolated.append(
            interpolate_z_stack(
                image=channel,
                pixelsize_in=pixelsize_in,
                pixelsize_out=pixelsize_out,
                method=method,
            )
        )
        if parent_ui:
            parent_ui.progress_update.emit({"value": i + 1, "max": image.shape[0]})
    return np.array(ch_interpolated)


def multi_channel_get_z_guass(image: np.ndarray, x: int, y: int, show: bool = False) -> List[float]:
    """Get the best fitting z-value for a 2D point in an multi-channel ZYX image using a 1D Gaussian fit:
    Args:
        x: x coordinate
        y: y coordinate
        image: 4D numpy array (CZYX)
        show: show the plot of the fit (for debugging)
    Returns:
        z: z coordinate (index)
    """
    # shortcut for single channel images
    if image.ndim == 3:
        return get_z_gauss(image, x, y, show=show)

    z_values = []
    for channel in image:
        z_values.append(get_z_gauss(channel, x, y, show=show))

    # get the channel with the maximum z-value (zval, zidx, zsigma)
    vals = np.array(z_values)
    ch_idx = np.argmax(vals[:, 0])

    return vals[ch_idx] # zval, zidx, zsigma

def zyx_targeting(
    img: np.ndarray,
    x: int,
    y: int,
    cutout: int = 15,
    apply_threshold: bool = False,
    threshold_val: float = 0.1,
    repeats: int = 5,
):
    print(f"Initial x: {x}, y: {y}")
    # get the initial z position (Note: this can fail, returns None)
    zval, zidx, zsigma = multi_channel_get_z_guass(image=img, x=x, y=y)

    assert img.ndim == 3, "Image must be 3D"

    for repeat in range(repeats):

        data = extract_image_patch(img, x, y, zidx, cutout)
        if data is None:
            break  # Or handle error case differently

        if apply_threshold: # apply threshold on normalized data
            data = threshold_image(data, threshold_val)
        
        # fit a 2d guassian to the 3d cutout
        poptXY = fitgaussian(data)
        if poptXY is None:
            break

        (height, xopt, yopt, width_x, width_y) = poptXY

        # x and y are switched when applying the offset
        x = int(x-cutout+yopt)
        y = int(y-cutout+xopt)
        width, height = img.shape[-1], img.shape[-2]
        if not (0 <= x < width and 0 <= y < height):
            break    

        # fit a 1d guassian to the z stack
        zval, zidx, zsigma = get_z_gauss(img, x=x, y=y)
        print(f"repeat: {repeat}, x: {x}, y: {y}, z: {zidx}, zval: {zval}, zsigma: {zsigma}")

    return x, y, (zval, zidx, zsigma)

def multi_channel_zyx_targeting(image: np.ndarray, xinit: int, yinit: int) -> Tuple[int, Tuple[int, int, int]]:
    """ZYX targeting for multi-channel images
    Args:
        image: 4D numpy array (CZYX)
        xinit: initial x coordinate
        yinit: initial y coordinate
    Returns:
        ch_idx: channel index with the best z-value
        x, y, z: x, y, z coordinates of the best z-value in the best channel
    """    

    # shortcut for single channel images
    if image.ndim == 3:
        x1, y1, (zv, z1, zs) = zyx_targeting(image, xinit, yinit)
        return 0, (x1, y1, z1)

    if image.ndim != 4:
        raise ValueError(f"image must be a 4D array (CZYX), got {image.ndim}")

    zvalues = []
    xyz_vals = []

    for i in range(image.shape[0]):
        ch_image = image[i]
        try:
            x1, y1, (zv, z1, zs) = zyx_targeting(ch_image, xinit, yinit)
        except Exception as e:
            print(f"An error occured: {e}")
            x1, y1, zv, z1, zs = xinit, yinit, 0, None, None
        zvalues.append((zv, z1, zs))
        xyz_vals.append((x1, y1, z1))

    vals = np.array(zvalues).astype(np.float32)
    ch_idx = np.argmax(vals[:, 0])
    print(f"Channel Index: {ch_idx}: xyz: {xyz_vals[ch_idx]}")

    return ch_idx, xyz_vals[ch_idx]
