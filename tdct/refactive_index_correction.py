import numpy as np
from typing import Tuple, List

# ref: https://github.com/hoogenboom-group/SF/blob/master/rescaling_of_microscopy_data/depth_dependent_rescaling.ipynb
# ref: https://opg.optica.org/optica/fulltext.cfm?uri=optica-11-4-553&id=549213
# ref: https://axialscaling.pythonanywhere.com/

def Lyakin(z,n_sample,n_im,NA): # https://doi.org/10.1134/S0030400X17090235
    d = 1
    top = np.add(n_im,np.sqrt(np.subtract(np.power(n_im,2),np.power(NA,2))))
    bottom_1 = np.multiply(4,np.subtract(np.power(n_sample,2),np.power(n_im,2)))
    bottom_2 = np.add(n_im,np.emath.sqrt(np.subtract(np.power(n_im,2),np.power(NA,2))))
    bottom = np.real(np.emath.sqrt(np.add(bottom_1,np.power(bottom_2,2))))
    if bottom == 0: bottom=0.000000000000001
    dz = np.multiply(d,np.divide(top,bottom))
    scaling_factor = np.divide(1,dz)    
    return np.zeros(len(z)) + scaling_factor

def scaling_factor_from_nfp(z: np.ndarray, NA: float, n1: float, n2: float, lam_0: float, crit: str = 'Lyakin'):
    #dzeta vs NFP
    n2overn1 = np.divide(n2,n1)
    m = np.emath.sqrt(np.subtract(np.power(n2,2),np.power(n1,2)))

    if n2overn1 < 1: delta = np.multiply(-1,np.divide(lam_0/4, np.multiply(n1,z)))
    else: delta = np.divide(lam_0/4, np.multiply(n1,z))
    one_plus_delta = np.add(1,delta)

    first_term = np.multiply(n2overn1,one_plus_delta)
    sec_term = np.multiply(np.divide(m,n1),np.emath.sqrt(np.multiply(delta,np.add(2,delta))))
    sf_univ = np.add(first_term, sec_term)

    sf = np.zeros(len(z))
    if crit != 'None': #cap off exploding SF for small depths
        if crit == 'Loginov': #use Loginov's critical value
            sf_crit = np.divide(n1-np.emath.sqrt(np.power(n1,2)-np.power(NA,2)),
                                    n2-np.emath.sqrt(np.power(n2,2)-np.power(NA,2)))
        elif crit == 'Lyakin': # use Lyakin/Stallinga's value
            sf_crit = Lyakin([0],n2,n1,NA)[0]
        for i in range(len(sf)):
            if n2overn1 < 1: sf[i] = np.max([np.real(sf_univ[i]),np.real(sf_crit)])
            elif n2overn1 > 1:sf[i] = np.min([np.real(sf_univ[i]),np.real(sf_crit)])
            else: sf[i]=1
    else: sf = np.real(sf_univ) #when no capping is performed
    return sf

def rescale_stack(stack: np.ndarray, NA: float, n1: float, n2: float, lam_0: float, ps_z: float, crit = 'Lyakin'):
    """
    Rescale a stack of images to correct for refractive index mismatch.
    Args:
        stack (np.ndarray): 3D stack of images (Z, Y, X).
        NA (float): Numerical aperture of the objective lens.
        n1 (float): Refractive index of the immersion medium.
        n2 (float): Refractive index of the sample.
        lam_0 (float): Wavelength of light in vacuum (in meters).
        ps_z (float): Pixel size in the z-direction (in meters).
        crit (str): Capping method for scaling factor ('Lyakin', 'Loginov', or 'None').
    Returns:
        np.ndarray: Rescaled stack of images.
        np.ndarray: New focal positions (AFP) of the rescaled stack (original pixelsize-z).
        np.ndarray: Actual focal positions (AFP) of the input stack.
        np.ndarray: Apparent Focal positions (NFP) of the input stack.
    """
    if stack.ndim != 3:
        raise ValueError("Input stack must be a 3D array (Z, Y, X).")

    nz, ny, nx = stack.shape

    # NFP: apparent focal positions
    # AFP: actual focal positions

    # make list of apparent focal positions (NFP) of input stack (with ref index mismatch)
    nfp_stack = np.arange(nz) * ps_z # zpos
    nfp_stack[0] = 1e-20 # prevent SF function from blowing up
    
    # calculate depth-dependent scaling factor for each NFP
    sf_stack = scaling_factor_from_nfp(z=nfp_stack, NA=NA, n1=n1, n2=n2, lam_0=lam_0, crit=crit)  # scaling factor, constant in region for lamella (~10um)
    nfp_stack[0] = 0.0 # make first value of NFP array zero again (funky, I know)
    
    # calculate actual focal position (AFP) of stack using scaling factor and NFP
    afp_stack = np.multiply(nfp_stack, sf_stack)

    # make new stack rescaled data will be added to
    # use step size of original stack and range calculated from depth-dependent scaling of last slide in input stack
    afp_new_stack = np.arange(0, afp_stack[-1], ps_z)

    # assumes z-stack, ZYX
    print(nfp_stack)    # z-pos (apparent)
    print(sf_stack)     # scaling factor
    print(afp_stack)    # scaled z-pos (actual)
    print(afp_new_stack)  # new z-pos, with original step size
    print(len(nfp_stack), len(afp_stack), len(afp_new_stack)) # length of input stack
    
    # make empty array we will fill with intensities of rescaled stack
    stack_rescaled = np.empty([afp_new_stack.shape[0], ny , nx])

    # first slice, no rescaling
    stack_rescaled[0] = stack[0]

    # put intensities of rescaled stack into new evenly spaced stack:
    for i in range(1, afp_new_stack.shape[0]):
        afp_slide = afp_new_stack[i] #get AFP of slice in new stack
        
        #find two nearest slices in mismatched stack and their AFPs
        index, value =min(enumerate(afp_stack), key=lambda x: abs(x[1]-afp_slide)) #get the index of the closest AFP value in the AFP list of the mismatched stack
        #get the indices of the slices in the mismatch stack surrounding the AFP value in the new stack       
        indices = [index, index+1] if afp_slide > value else [index-1, index]

        #get the corresponding AFP values in the mismatched stack
        value_under, value_upper = afp_stack[indices[0]], afp_stack[indices[1]]
        #calculate the AFP distance between the slice in the new stack and the surrounding slices in the mismatched stack
        dz_under, dz_upper = afp_slide - value_under, value_upper - afp_slide
        
        # we will use inverse distance weighting with power of 1 to interpolate the intensity in the new stack: https://en.wikipedia.org/wiki/Inverse_distance_weighting :
        dz_under_inv, dz_upper_inv=1/dz_under, 1/dz_upper
        dz_sum = dz_under + dz_upper
        dz_inv_sum = dz_under_inv + dz_upper_inv
        # make interpolate intensities in new slide from the two closest slices in the mismatched stack:
        new_slide = np.divide(
            np.multiply(stack[indices[0],:,:], dz_under_inv) + 
            np.multiply(stack[indices[1],:,:], dz_upper_inv), 
                              dz_inv_sum)

        #save to new stack
        stack_rescaled[i] = new_slide
        
        #debug
        # print('AFP new stack (um) ', afp_slide,# 'Mean intensity slide: ', np.mean(new_slide)*10000,
        #   '\nAFP under old stack', value_under,# 'Mean intensity slide under: ', np.mean(stack[indices[0],:,:])*10000,
        #       '\nAFP upper old stack',value_upper,# 'Mean intensity slide upper: ', np.mean(stack[indices[0],:,:])*10000,
        #       '\n' )
        
        # #plot new slices and surrounding slices
        # fig,axs=plt.subplots(1,3)
        # fig.set_figheight(10)
        # fig.set_figwidth(20)
        # axs[0].imshow(new_slide)
        # axs[0].set_title('New stack index: '+str(i))
        # axs[1].set_title('Old stack index under: '+str(indices[0]))
        # axs[2].set_title('Old stack index upper: '+str(indices[1]))
        # axs[1].imshow(stack[indices[0],:,:])
        # axs[2].imshow(stack[indices[1],:,:])
        # plt.show()
        
        #print indices in both stacks
        # print('\nNew stack index: '+str(i))
        # print('Old stack index under: '+str(indices[0]))
        # print('Old stack index upper: '+str(indices[1]))
            
    return stack_rescaled, afp_new_stack, afp_stack, nfp_stack

# correct correlation targets
def calculate_correction(pos_target: Tuple[int, int], 
                         pos_edge: Tuple[int, int], 
                         milling_angle: float, 
                         scale_factor) -> Tuple[float, float]:

    # positions in image pixels
    # milling angle in radians

    pos_target = np.array(pos_target)
    pos_edge = np.array(pos_edge)

    # pos
    dpx = pos_edge - pos_target

    # correct for apparent view and scale factor
    dview = dpx/np.cos(milling_angle) * scale_factor

    return dview # y, x

def apply_correction_to_targets(targets: List[Tuple[int, int]], correction: Tuple[float, float]) -> List[Tuple[int, int]]:
    """
    Apply the correction to the target positions.
    Args:
        targets (List[Tuple[int, int]]): List of target positions.
        correction (Tuple[float, float]): Correction values for y and x.
    Returns:
        List[Tuple[int, int]]: List of corrected target positions.
    """
    corrected_targets = []
    for target in targets:
        corrected_target = (target[0] + correction[0], target[1] + correction[1])
        corrected_targets.append(corrected_target)
    return corrected_targets