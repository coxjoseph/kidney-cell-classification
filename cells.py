import matplotlib.pyplot as plt
import numpy as np
import cv2
import multiprocessing
from math import sqrt
from dataclasses import dataclass
from typing import Union
from logging import getLogger
from skimage.filters import threshold_isodata, threshold_otsu
from skimage import io, transform
from feature_extraction import get_nuclei_size

logger = getLogger()

@dataclass
class Nucleus:
    center: tuple # Nucleus center in (m,n) coordinate space
    pixel_area: int # Number of ON pixels in the binary mask

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

# Calculates and returns a single small-window nucleus mask from the original CODEX DAPI layer.
def get_nucleus_mask_dapi(nucleus_coordinates, codex: np.ndarray, DAPI_index, global_threshold, window_size=256, erosion_radius=2.5, isolated=True, visual_output=False) -> np.ndarray:
    # Get bounding box coordinates
    upper_m, lower_m, left_n, right_n = get_bounding_box(nucleus_coordinates, window_size, codex_shape=codex.shape)
    
    # Get slice of DAPI CODEX around the target coordinates
    subset_indices = (DAPI_index, slice(upper_m, lower_m), slice(left_n, right_n))
    nuclei_mask = codex[subset_indices]
        
    # Threshold nucleus stain using the higher of the global and local threshold values
    # Setting a minimum threshold of the global prevents erroneous detection of nuclei in regions that contain zero nuclei
    threshold = max(global_threshold, threshold_otsu(nuclei_mask)) 
    nuclei_mask = (nuclei_mask > threshold).astype(np.uint8) # Binarizes the image
    
    if visual_output:
        plt.figure(figsize=(10, 8))
        plt.imshow(nuclei_mask, cmap='hot')
        plt.title(f'Nuclei Mask Before Distance Transform')
        plt.colorbar()
        plt.show()
    
    # Perform distance transformation
    threshold_scaler = 1.5
    dist_transform = cv2.distanceTransform(nuclei_mask, cv2.DIST_L2, 3)
    threshold = threshold_otsu(dist_transform)*threshold_scaler # Automatically obtain a threshold value
    dist_transform = (dist_transform > threshold).astype(np.uint8) # Binarizes the image
    nuclei_mask = dist_transform
    
    if visual_output:
        plt.figure(figsize=(10, 8))
        plt.imshow(nuclei_mask, cmap='hot')
        plt.title(f'Small Window Nuclei Mask Before Erosion, After Distance Transform')
        plt.colorbar()
        plt.show()
    
    # Perform erosion to remove very small nuclei
    kernel = make_circular_kernel(erosion_radius)
    nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_ERODE, kernel)
    
    if visual_output:
        plt.figure(figsize=(10, 8))
        plt.imshow(nuclei_mask, cmap='hot')
        plt.title(f'Small Window Nuclei Mask After Erosion')
        plt.colorbar()
        plt.show()
    
    # Erase all neighboring, disconnected nuclei from the mask
    if isolated:
        nuclei_mask = isolate_nuclei(nuclei_mask, erosion_radius, visual_output)
    
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
# Expects an already downsampled mask to be passed in
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
                coordinates_rescaled = (m*downsample_factor + int(downsample_factor/2),n*downsample_factor + int(downsample_factor/2)) # Convert coordinates back to original coordinate space
                # Nucleus size instantiated to -1 to indicate it has not been calculated yet
                # Since only the downsampled mask is passed here, it is impossible to get an accurate size estimate
                nuc = Nucleus(coordinates_rescaled, pixel_area=-1) 
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

def segment_nuclei_brightfield(brightfield: np.ndarray, window_size=512, visual_output=True) -> np.ndarray:
    print('Segmenting brightfield nuclei...', flush=True)
    
    num_m_iterations = int(brightfield.shape[0] / window_size)
    num_n_iterations = int(brightfield.shape[1] / window_size)

    # Allocate result array
    whole_image_mask = np.empty((num_m_iterations * window_size, num_n_iterations * window_size), dtype=np.uint8)

    # Convert image to float32
    brightfield_adjusted = brightfield.astype(np.float32) / 255.0
    
    # Kernels for morphological operations
    kernel = np.ones((5, 5), np.uint8)
    closing_kernel = np.ones((3,3), np.uint8)
    
    # Kernels for morphological operations
    dilation_kernel = np.ones((5, 5), np.uint8)
    opening_kernel = np.ones((3,3), np.uint8)
    erosion_kernel = np.ones((7,7), np.uint8)
    closing_kernel = np.ones((3,3), np.uint8)
    
    # Process tiles
    for m in range(num_m_iterations):
        for n in range(num_n_iterations):
            r = m * window_size
            c = n * window_size
            
            # Extract the tile
            tile = brightfield_adjusted[r:r+window_size, c:c+window_size, :]

            # Color deconvolution
            stain_matrix = np.array([[0.711, 0.618, 0.336],
                                     [0.596, 0.645, 0.478],
                                     [0.43, -0.76, 0.488]])
                                     
            # Perform color deconvolution on each tile
            stains = np.dot(-np.log(tile + 1e-9), np.linalg.inv(stain_matrix))

            # Handle negative values
            stains[stains < 0] = 0  

            # Normalize the stain images
            stains_max = np.max(stains, axis=(0, 1))
            stains_max_normalized = stains_max.copy()
            stains_max_normalized[stains_max_normalized == 0] = 1  # Avoid division by zero
            stains_normalized = np.divide(stains, stains_max_normalized[np.newaxis, np.newaxis, :], 
                                          out=np.zeros_like(stains), where=stains_max_normalized != 0)

            hematoxylin_channel = stains[:, :, 0]
            
            # Replace NaN values with a predefined fill value (e.g., 0)
            hematoxylin_channel_filled = np.nan_to_num(hematoxylin_channel, nan=0)

            # Compute the range of valid values, excluding NaN
            valid_min = np.min(hematoxylin_channel_filled)
            valid_max = np.max(hematoxylin_channel_filled)

            # Scale the hematoxylin channel to the range [0, 255], handling NaN values
            hematoxylin_channel_grayscale = ((hematoxylin_channel_filled - valid_min) / (valid_max - valid_min) * 255).astype(np.uint8)

            # Thresholding
            binary_mask =  hematoxylin_channel_grayscale > threshold_otsu(hematoxylin_channel_grayscale)

            # Resize binary mask to match window size
            binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8), (window_size, window_size))

            # Closing to fill small holes inside the objects and connect nearby objects
            binary_mask_resized = cv2.morphologyEx(binary_mask_resized, cv2.MORPH_CLOSE, closing_kernel)

            # Distance transform
            distance_transform = ndimage.distance_transform_edt(binary_mask_resized)
            
            # Find markers using morphological operations
            _, markers = cv2.connectedComponents(binary_mask_resized)

            # Perform watershed segmentation
            labels = watershed(-distance_transform, markers, mask=binary_mask_resized)
            
            # Threshold the segmented labels to obtain a binary image
            binary_labels = labels.astype(np.uint8)
            binary_labels[binary_labels > 0] = 1
            
            # Resize binary mask to match window size
            binary_mask = cv2.resize(binary_labels.astype(np.uint8), (window_size, window_size))

            # Morphological operations: 
            # Erosion to remove small bright regions and detach connected components
            binary_mask = cv2.erode(binary_mask, erosion_kernel, iterations=1)

            # Dilation to restore the original size of the objects while keeping them separated
            binary_mask = cv2.dilate(binary_mask, dilation_kernel, iterations=1)

            # Opening to remove small bright spots while preserving the shape of larger objects
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, opening_kernel)

            # Closing to fill small holes inside the objects and connect nearby objects
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel)

            # Find contours of the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a black image to draw the filled polygons
            filled_mask = np.zeros_like(binary_mask, dtype=np.uint8)

            # Draw filled polygons for each contour
            cv2.fillPoly(filled_mask, contours, color=255)  # Filling with 255 (white) for binary representation

            # Update the whole image mask
            whole_image_mask[r:r+window_size, c:c+window_size] = filled_mask.astype(bool).astype(np.uint8)  # Convert back to uint8
            
            # Visualization for a specific tile (e.g., 10th row and 10th column)
            if m == 10 and n == 10:
                plt.imshow(hematoxylin_channel)
                plt.title('Color Deconvolution Tile 10,10')
                plt.axis('off')
                plt.show()
                plt.imshow(hematoxylin_channel_grayscale, cmap='gray')
                plt.title('Grayscale Tile 10,10')
                plt.axis('off')
                plt.show()
                plt.imshow(filled_mask, cmap='gray')
                plt.title('Binary Mask Tile 10,10')
                plt.axis('off')
                plt.show()
                
                
    if visual_output:
        center_tile_index_row = num_m_iterations // 3
        center_tile_index_col = num_n_iterations // 3
        plt.imshow(whole_image_mask[center_tile_index_row * window_size:(center_tile_index_row + 1) * window_size,
                                      center_tile_index_col * window_size:(center_tile_index_col + 1) * window_size],
                   cmap='gray')
        plt.title('Binary Mask (Center Tile) from H&E Segmentation')
        plt.axis('off')
        plt.show()

        # Plot segmented nuclei (you may need to adjust this line based on your implementation)
        plt.imshow(whole_image_mask, cmap='gray')
        plt.axis('off')
        plt.show()

    return whole_image_mask
    
def segment_nuclei_dapi(codex: np.ndarray, DAPI_index=0, erosion_radius=2.5, window_size=256, visual_output=False) -> np.ndarray:
    # Build a whole-image nuclei segmentation by doing piecewise small window segmentations of the DAPI image and merging the results. If the image dimensions are not # multiples of the window size, the image will be cropped to the nearest multiple.
    print('Segmenting DAPI nuclei...', flush=True)
    
    num_m_iterations = int(codex.shape[1]/window_size)
    num_n_iterations = int(codex.shape[2]/window_size)
    
    # Allocate result array
    whole_image_mask = np.empty((num_m_iterations*window_size, num_n_iterations*window_size),dtype=np.uint8)
    global_threshold = threshold_otsu(codex[DAPI_index])
    
    for m in range(0, num_m_iterations):
        for n in range(0, num_n_iterations):
            # Calculate the box boundaries for the current window
            box_center = (m*window_size + window_size/2, n*window_size + window_size/2)
    
            # Get a local (small window) nuclei mask
            local_mask = get_nucleus_mask_dapi(box_center, codex, DAPI_index, global_threshold=global_threshold, window_size=window_size, erosion_radius=erosion_radius, isolated=False, visual_output=False)
            
            # Copy local nuclei mask to the global mask
            whole_image_mask[m*window_size:(m+1)*window_size, n*window_size:(n+1)*window_size] = local_mask
    return whole_image_mask
           

# Mask out all nuclei not connected to the nuclei at the center of the window
def isolate_nuclei(nuclei_window: np.ndarray, erosion_radius, visual_output=False) -> np.ndarray:
    isolated_window = nuclei_window.copy()
    primary_nuclei_marker = 2 # Arbitrary value to separate the center nuclei from other nuclei visible in the window
    center = int(nuclei_window.shape[0]/2)
    cv2.floodFill(isolated_window, None, (center,center), primary_nuclei_marker) # Give all connected pixels the primary nuclei marker
    isolated_window = (isolated_window == primary_nuclei_marker).astype(np.uint8)
    
    # Perform dilation to restore the nucleus to its pre-erosion size
    kernel = make_circular_kernel(erosion_radius)
    isolated_window = cv2.morphologyEx(isolated_window, cv2.MORPH_DILATE, kernel)
    
    if visual_output:
        plt.figure(figsize=(4, 4))
        plt.imshow(isolated_window, cmap='hot')
        plt.title(f'Isolated Nuclei')
        plt.colorbar()
        plt.show()
    
    return isolated_window

def calculate_radii_from_nuclei(nuclei, nuclei_mask, window_size=256) -> list[int]:
    radii = []
    print(f"Calculating radii from nuclei list...", flush=True) # Flush to force immediate output 
    for nucleus in nuclei:
        nucleus_mask = slice_nucleus_window(nuclei_mask, center_coordinates=nucleus.center, window_size=window_size) # Grab the local nuclei window
        _, labels = cv2.connectedComponents(nucleus_mask.astype(np.uint8)) # Count the number of nuclei
        num_nearby_nuclei = max(np.max(labels), 1) # Floor 1 to prevent divison by zero
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

def get_bounding_box(nucleus_coordinates, window_size, codex_shape) -> list[int]:        
        # Determine a bounding box for a small window around a target nucleus
        upper_m = int(nucleus_coordinates[0]-window_size/2)
        lower_m = int(nucleus_coordinates[0]+window_size/2)
        left_n = int(nucleus_coordinates[1]-window_size/2)
        right_n = int(nucleus_coordinates[1]+window_size/2)

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
        
def slice_nucleus_window(nuclei_mask: np.ndarray, center_coordinates, window_size) -> np.ndarray:
    # Grab the relevant local nucleus segmentation from a larger nuclei mask
    codex_shape = (1, nuclei_mask.shape[0], nuclei_mask.shape[1]) # CODEX isn't passed here, but only second two dimensions are needed
    upper_m, lower_m, left_n, right_n = get_bounding_box(center_coordinates, window_size, codex_shape=codex_shape)
    nucleus_mask = nuclei_mask[upper_m:lower_m, left_n:right_n] # Grab the relevant window of the already segmented nucleus mask
    return nucleus_mask
    
# Removes a percentage of the largest nuclei from a binary mask
# Intent is to improve H&E segmentation by removing large non-cell artifacts (such as slice boundaries) from the segmentation
def remove_largest_nuclei(nuclei: list[Nucleus], nuclei_mask: np.ndarray, cull_percent=0.05, visual_output=False) -> np.ndarray:
    print('Removing large outlier nuclei...')
    
    if (visual_output):
        # Output not shown until after computation is complete
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(nuclei_mask, cmap='gray')
        plt.title('Binary Mask before Culling Large Nuclei')
        plt.axis('off')
    
    # Pull a numpy array of sizes from the nuclei list
    num_nuclei = len(nuclei)
    pixel_areas = np.empty(num_nuclei, dtype=int)
    for n in range(num_nuclei):
        pixel_areas[n] = nuclei[n].pixel_area
        
    hist, bin_edges = np.histogram(pixel_areas, range=(0,255), density=True) # Obtain the probability density function
    cdf = np.cumsum(hist) # Obtain the cumulative density function
    
    # Find an area threshold
    cdf_cutoff = 1 - cull_percent
    index = np.argmax(cdf >= cdf_cutoff) # Gets the first index of where the cdf is greater than the cutoff
    threshold = bin_edges[index]
    
    # Iterate through the list of nuclei and mask out those that have a size greater than the threshold
    for nuc in nuclei:
        if (nuc.pixel_area >= threshold):
            m = nuc.center[0]
            n = nuc.center[1]
            cv2.floodFill(nuclei_mask, None, (n,m), 0)# Remove the found nucleus from the mask. OpenCV uses (x,y) coordinate space
            
    if (visual_output):
        plt.subplot(1, 2, 2)
        plt.imshow(nuclei_mask, cmap='gray')
        plt.title('Binary Mask after Culling Large Nuclei')
        plt.axis('off')
        plt.show()
        
    return nuclei_mask
    
# Takes in a list of Nucleus objects, a global binary nucleus mask, and the maximum window size around each nucleus
# Returns a list of Nuclei with their size parameters corrected
def calculate_nuclei_sizes(nuclei: list[Nucleus], nuclei_mask, window_size=128) -> list[Nucleus]:
    print('Calculating nuclei sizes...', flush=True)
    debug_count = 0
    for nuc in nuclei:
        nucleus_mask = slice_nucleus_window(nuclei_mask, nuc.center, window_size)
        nucleus_mask = isolate_nuclei(nucleus_mask, erosion_radius=2.5)
        pixel_area = get_nuclei_size(nucleus_mask)
        nuc.pixel_area = pixel_area
    return nuclei
