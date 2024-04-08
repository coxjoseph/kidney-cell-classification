from os import PathLike
from typing import Union
from skimage import io, transform
from cells import get_nucleus_mask, get_bounding_box
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile


def load_tiff_as_array(path: Union[str, PathLike]) -> np.ndarray:
    """
    Load a TIFF image and convert it to a numpy array.

    Parameters:
    - file_path: Path to the TIFF file.

    Returns:
    - A numpy array representing the image.
    """
    with tifffile.TiffFile(path) as tiff:
        array = tiff.asarray()
    return array


def display_channel_heatmaps(array: np.ndarray, indices: list) -> None:
    """
    Display specified channels of the image as heatmaps.

    Parameters:
    - array: A numpy array representing the image.
    - indices: A list of indices of the channels to be displayed.
    """
    for index in indices:
        plt.figure(figsize=(10, 8))
        plt.imshow(image_array[index, :, :], cmap='hot')
        plt.title(f'Channel {index} Heatmap')
        plt.colorbar()
        plt.show()
        
def overlay_nuclei_boundaries(nucleus_coordinates, codex: np.ndarray, DAPI_index, mask_size=256) -> None:
    """
    Display the DAPI CODEX layer with the detected nuclei edges overlayed.
    
    Parameters:
    - nucleus_coordinates: Tuple of (m,n) coordinates for the center of the nucleus window
    - codex: (HxMxN) CODEX Tiff data
    - DAPI index: Layer of the CODEX file in which DAPI data is stored
    - mask_size: Size of the window around the target nucleus
    """
    nuclei_mask = get_nucleus_mask(nucleus_coordinates, codex, DAPI_index, mask_size=mask_size, isolated=False)
    
    # Perform edge detection using erosion
    structure = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    boundaries = cv2.morphologyEx(nuclei_mask, cv2.MORPH_GRADIENT, structure)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(nuclei_mask)
    plt.title('nuclei_mask')
    plt.show()
    
    
    plt.figure(figsize=(10, 8))
    plt.imshow(boundaries)
    plt.title('Nuclei Boundary')
    plt.show()
    
        
    # Grab the CODEX slice
    upper_m, lower_m, left_n, right_n = get_bounding_box(nucleus_coordinates, mask_size, codex_shape=codex.shape)
    subset_indices = (DAPI_index, slice(upper_m, lower_m), slice(left_n, right_n))
    codex_slice = codex[subset_indices]
    
    nuclei_mask_rgb = np.zeros((mask_size, mask_size, 3), dtype=np.uint8)
    nuclei_mask_rgb[:,:,0] = codex_slice
    #nuclei_mask_rgb[:,:,0] = nuclei_mask_rgb[:,:,0] * ~boundaries # Prevents color smear with boundaries overlay
    nuclei_mask_rgb[:,:,1] = boundaries*255

    
    #nuclei_mask_rgb[:,:,2] = codex[DAPI_index].copy()
    
    # Display
    plt.figure(figsize=(10, 8))
    plt.imshow(nuclei_mask_rgb)
    plt.title('Cell Boundary')
    plt.show()
        
def overlay_cell_boundaries(nuclei_mask: np.ndarray, radius) -> None:
    """
    Draw a green circle of the cell radius over its nuclei mask.

    Parameters:
    - nuclei_mask: A square binary mask centered around the target nuclei.
    - radius: Radius of the cell.
    """

    # Convert binary image to RGB
    m, n = nuclei_mask.shape
    nuclei_mask_rgb = np.empty((m, n, 3), dtype=np.uint8)
    nuclei_mask_rgb[:,:,0] = nuclei_mask.copy()
    nuclei_mask_rgb[:,:,1] = nuclei_mask.copy()
    nuclei_mask_rgb[:,:,2] = nuclei_mask.copy()
    
    center = nuclei_mask.shape[0]/2
    
    # Restore brightness
    for m in range(nuclei_mask_rgb.shape[0]):
        for n in range(nuclei_mask_rgb.shape[1]):
            for channel in range(nuclei_mask_rgb.shape[2]):
                nuclei_mask_rgb[m][n][channel] = nuclei_mask_rgb[m][n][channel] * 255
                
            tol = 5
            if abs(int((m-center)**2 + (n-center)**2) - int(radius**2)) < tol:
                nuclei_mask_rgb[m][n][1] = 255
                
    # Display
    plt.figure(figsize=(10, 8))
    plt.imshow(nuclei_mask_rgb)
    plt.title('Cell Boundary')
    plt.show()
    

if __name__ == '__main__':
    file_path = 'data/Section6_CODEX.tif'
    image_array = load_tiff_as_array(file_path)

    print(f'Image array is {"x".join(map(str, image_array.shape))}')
    channel_indices = np.linspace(0,40,dtype=int)
    display_channel_heatmaps(image_array, channel_indices)
