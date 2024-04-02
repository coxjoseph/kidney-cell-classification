import matplotlib.pyplot as plt
import numpy as np
import cv2
import multiprocessing
from math import sqrt
from dataclasses import dataclass
from typing import Union
from logging import getLogger
from skimage.filters import threshold_isodata
from skimage import io, transform

logger = getLogger()

@dataclass
class Nucleus:
    center: tuple


@dataclass
class Cell:
    nucleus: Nucleus
    radius: int
    features: Union[tuple, None] = None
    label: Union[int, None] = None

    # Get pixel values, assuming circular cell with center nuclei
    def get_pixel_values(self, codex: np.ndarray) -> np.ndarray:
        r = self.radius

        # Bounding square for circular nucleus.
        center_x, center_y = self.nucleus.center
        start_x, end_x = max(center_x + r, 0), min(center_x + r, codex.shape[0])
        start_y, end_y = max(center_y - r, 0), min(center_y + r, codex.shape[1])

        output_array = np.zeros(2 * r, 2 * r, codex.shape[2])

        # TODO: will this error if nucleus is too close to edge? unsure.
        x, y = np.ogrid[-r:r, -r:r]
        distance_squared = x ** 2 + y ** 2
        mask = distance_squared <= (r ** 2)

        # Copy the relevant parts of each channel in codex
        for i in range(codex.shape[2]):
            cropped_channel = codex[start_x:end_x, start_y:end_y, i]
            # Make sure the cropped_channel fits (I think this covers edge cases mentioned above)
            min_x, min_y = max(r - center_x, 0), max(r - center_y, 0)
            max_x, max_y = min_x + cropped_channel.shape[0], min_y + cropped_channel.shape[1]
            output_array[min_x:max_x, min_y:max_y, i] = cropped_channel

        # Zero out pixels outside center
        for i in range(output_array.shape[0]):
            output_array[i, ~mask] = 0

        logger.debug(f'DEBUG: {output_array.shape=}')
        logger.info('Channel-wise pixel values separated near nuclei...')
        return output_array

    def calculate_features(self, feature_extractors: list[callable], codex: np.ndarray) -> None:
        pixel_values = self.get_pixel_values(codex)  # THESE WILL BE SQUARE ARRAYS
        features = []
        [features.extend(feature_extractor(pixel_values)) for feature_extractor in feature_extractors]
        logger.info('Created features...')
        logger.debug(f'DEBUG: {features}')
        self.features = tuple(features)
        
def get_bounding_box(nucleus_coordinates, mask_size, codex_shape) -> list[int]:        
        # Determine a bounding box for the image
        upper_m = int(nucleus_coordinates[0]-mask_size/2)
        lower_m = int(nucleus_coordinates[0]+mask_size/2)
        left_n = int(nucleus_coordinates[1]-mask_size/2)
        right_n = int(nucleus_coordinates[1]+mask_size/2)

        # If the bounding box goes outside of the image, shift it such that it is inside of the image
        if (left_n < 0):
            right_n = right_n - left_n
            left_n = 0
        if (right_n >= codex_shape[2]):
            left_n = left_n - (right_n - codex_shape[2]) - 1
            right_n = codex_shape[2]-1
        if (upper_m < 0):
            lower_m = lower_m - upper_m
            upper_m = 0
        if (lower_m >= codex_shape[1]):
            upper_m = upper_m - (lower_m - codex_shape[1]) - 1
            lower_m = codex_shape[1]-1
        return [upper_m, lower_m, left_n, right_n]

def get_nucleus_mask(nucleus_coordinates, codex: np.ndarray, DAPI_index, mask_size=256, opening_radius=2.5, isolated=True, visual_output=False) -> np.ndarray:
    # Get bounding box coordinates
    upper_m, lower_m, left_n, right_n = get_bounding_box(nucleus_coordinates, mask_size, codex_shape=codex.shape)
        
    # Threshold nucleus stain
    subset_indices = (DAPI_index, slice(upper_m, lower_m), slice(left_n, right_n))
    nuclei_mask = codex[subset_indices]
    
    threshold_scaler = 1.1 # Manual adjustment factor for automatic threshold
    threshold = threshold_isodata(nuclei_mask)*threshold_scaler # Automatically obtain a threshold value
    nuclei_mask = (nuclei_mask > threshold).astype(np.uint8) # Binarizes the image
    
    if visual_output:
        plt.figure(figsize=(10, 8))
        plt.imshow(nuclei_mask, cmap='hot')
        plt.title(f'Small Window Nuclei Mask Before Erosion')
        plt.colorbar()
        plt.show()
    
    # Perform erosion to remove very small nuclei
    kernel = make_circular_kernel(opening_radius)
    nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_ERODE, kernel)
    
    if visual_output:
        plt.figure(figsize=(10, 8))
        plt.imshow(nuclei_mask, cmap='hot')
        plt.title(f'Small Window Nuclei Mask After Erosion')
        plt.colorbar()
        plt.show()
    
    # Erase all neighboring, disconnected nuclei from the mask
    if isolated:
        nuclei_mask = isolate_nuclei(nuclei_mask, opening_radius, visual_output)
    
    return nuclei_mask

# Wrapper function for parallel nuclei coordinate extraction
# CODEX is downsampled prior to reduce execution time
# If a small input subimage is used, decrease the process count to minimize boundary artifacts (duplicate nuclei)
# Visual output will show downsampled image
def extract_nuclei_coordinates(nuclei_mask: np.ndarray, downsample_factor=2, num_processes=8, visual_output=False)-> list[Nucleus]:
    scale_factor = 1/downsample_factor
    downsampled_mask = cv2.resize(nuclei_mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    if visual_output:
        plt.figure(figsize=(10, 8))
        plt.imshow(downsampled_mask, cmap='hot')
        plt.title(f'Downsampled mask')
        plt.colorbar()
        plt.show()
    
    print('Beginning nuclei coordinate extraction...', flush=True) # Flush to force immediate output 
    pool = multiprocessing.Pool()
    results = []
    for process_ID in range(0, num_processes):
        sublist = pool.apply_async(extract_nuclei_coordinates_parallel, (downsampled_mask, num_processes, process_ID, downsample_factor))
        results.append(sublist)
    
    # Wait for processes to finish
    pool.close()
    pool.join()
    
    # Get the nuclei coordinate results
    nuclei_list = []
    for result in results:
        nuclei_list.extend(result.get())   
        
    nucleus_count = len(nuclei_list)
    print(f"Total nucleus count: {nucleus_count}", flush=True) # Flush to force immediate output 
    return nuclei_list
    
# Individual process implementation for extracting nuclei coordinates from a slice of the image
# Image is sliced into groups of rows based on process ID
# Boundary conditions will cause double counting of some nuclei along the slice edges
def extract_nuclei_coordinates_parallel(nuclei_mask: np.ndarray, num_processes, process_ID, downsample_factor, verbose=True)-> list[Nucleus]:
    # Image is divided evenly along its rows for each process
    # Determines which block of rows this process operates on
    start_row = int((nuclei_mask.shape[0] / num_processes) * process_ID);
    end_row = int(((nuclei_mask.shape[0] / num_processes)) * (process_ID+1));
    
    nuclei_list = []
    count = 0
   
    # Scan through the image until a nuclei pixel is encountered
    for m in range(start_row, end_row):
        for n in range(0, nuclei_mask.shape[1]):
            # If a pixel is hit, append it to the nuclei list and set all connected pixels to zero
            if nuclei_mask[m][n]:
                nuc = Nucleus((m*downsample_factor + int(downsample_factor/2),n*downsample_factor + int(downsample_factor/2)))
                nuclei_list.append(nuc)
                cv2.floodFill(nuclei_mask, None, (n,m), 0)# Remove the found nucleus from the mask
                count = count + 1
                if (count % 500 == 0): # Periodic progress update
                    print(f"Current nucleus count (within process {process_ID}): {count}", flush=True) # Flush to force immediate output 
    return nuclei_list
    
def make_circular_kernel(kern_radius) -> np.ndarray:
    center = kern_radius
    x, y = np.ogrid[:2*kern_radius, :2*kern_radius]
    kernel = ((x - center) ** 2 + (y - center) ** 2 <= kern_radius ** 2).astype(np.uint8)
    return kernel

def segment_nuclei_brightfield(brightfield: np.ndarray) -> list[Nucleus]:
    # Example for now:
    example_nuclei = [Nucleus(center=(x, y)) for x, y in zip(range(5), range(5))]
    logger.info(f'Segmented {len(example_nuclei)} nuclei')
    # TODO: create a method/methods that look at the brightfield array (should we switch to the codex image?) and
    #  create a list of center-points for nuclei. Once we have that uncomment below:

    # centers = get_center_points(brightfield)
    # logger.info(f'Segmented {len(centers)} nuclei')
    # return [Nucleus(center=center) for center in centers]
    return example_nuclei
    
# Returns a binary mask of all nuclei from the CODEX
def segment_nuclei_dapi(codex: np.ndarray, DAPI_index, erosion_radius=2.5, visual_output=False) -> np.ndarray:
    nuclei_mask = codex[DAPI_index]
    threshold = threshold_isodata(nuclei_mask) # Automatically obtain a threshold value
    nuclei_mask = (nuclei_mask > threshold).astype(np.uint8) # Binarizes the image
    kernel = make_circular_kernel(erosion_radius)
    nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_ERODE, kernel)
    
    if visual_output:
        plt.figure(figsize=(10, 8))
        plt.imshow(nuclei_mask, cmap='hot', interpolation=None)
        plt.title(f'DAPI Nuclei Mask')
        plt.colorbar()
        plt.show()
 
    return nuclei_mask

# Mask out all nuclei not connected to the nuclei at the center of the window
def isolate_nuclei(nuclei_window: np.ndarray, opening_radius, visual_output=False) -> np.ndarray:
    isolated_window = nuclei_window.copy()
    primary_nuclei_marker = 2 # Arbitrary value to separate the center nuclei from other nuclei visible in the window
    center = int(nuclei_window.shape[0]/2)
    cv2.floodFill(isolated_window, None, (center,center), primary_nuclei_marker) # Give all connected pixels the primary nuclei marker
    isolated_window = (isolated_window == primary_nuclei_marker).astype(np.uint8)
    
    # Perform dilation to restore the nucleus to its pre-erosion size
    kernel = make_circular_kernel(opening_radius)
    isolated_window = cv2.morphologyEx(isolated_window, cv2.MORPH_DILATE, kernel)
    
    if visual_output:
        plt.figure(figsize=(4, 4))
        plt.imshow(isolated_window, cmap='hot')
        plt.title(f'Isolated Nuclei')
        plt.colorbar()
        plt.show()
    
    return isolated_window

def calculate_radii_from_nuclei(nuclei, codex: np.ndarray, DAPI_index, window_size=256) -> list[int]:
    radii = []
    print(f"Calculating radii from nuclei list...", flush=True) # Flush to force immediate output 
    for nucleus in nuclei:
        nucleus_mask = get_nucleus_mask(nucleus.center, codex, DAPI_index, mask_size=window_size, isolated=False)
        _, labels = cv2.connectedComponents(nucleus_mask.astype(np.uint8))
        num_nearby_nuclei = np.max(labels)
        average_cell_area = (window_size*window_size)/num_nearby_nuclei
        radius = sqrt(average_cell_area/3.14)
        radii.append(radius)
        
    return radii


def create_cells(nuclei: list[Nucleus], radii: list[int]) -> list[Cell]:
    if not len(nuclei) == len(radii):
        raise ValueError('Cannot initialize Cell objects: Radii and nuclei have different lengths: '
                         f'({len(nuclei)=} | {len(radii)=}).')

    return [Cell(nucleus=nucleus, radius=radius) for nucleus, radius in zip(nuclei, radii)]

def create_nuclei(nuclei_coordinates) -> list[Nucleus]:
    nuclei = []
    for nuc in nuclei_coordinates:
        nuclei.append(Nucleus(center=nuc))
    return nuclei
