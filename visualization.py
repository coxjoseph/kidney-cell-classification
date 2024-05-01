from os import PathLike
from typing import Union
from skimage import io, transform
from cells import get_nucleus_mask_dapi, get_bounding_box, slice_nucleus_window
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import seaborn as sns
from cells import Nucleus
import wsi_annotations_kit.wsi_annotations_kit as wak
from shapely.geometry import Point
import random as rd
import cv2
import json
import lxml.etree as ET
import uuid


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
        
def overlay_nuclei_boundaries(nucleus_coordinates, nuclei_mask, codex: np.ndarray, DAPI_index, mask_size=256) -> None:
    """
    Display the DAPI CODEX layer with the detected nuclei edges overlayed.
    
    Parameters:
    - nucleus_coordinates: Tuple of (m,n) coordinates for the center of the nucleus window
    - codex: (HxMxN) CODEX Tiff data
    - DAPI index: Layer of the CODEX file in which DAPI data is stored
    - mask_size: Size of the window around the target nucleus
    """
    local_nuclei_mask = slice_nucleus_window(nuclei_mask, nucleus_coordinates, window_size=mask_size)
    
    # Perform edge detection using erosion
    structure = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    boundaries = cv2.morphologyEx(local_nuclei_mask, cv2.MORPH_GRADIENT, structure)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(local_nuclei_mask)
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
    plt.title(f'Nuclei Boundaries at {nucleus_coordinates}')
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
                
            tol = 10 # Determines the thickness of the line
            if abs(int((m-center)**2 + (n-center)**2) - int(radius**2)) < tol:
                nuclei_mask_rgb[m][n][1] = 255
                
    # Display
    plt.figure(figsize=(10, 8))
    plt.imshow(nuclei_mask_rgb)
    plt.title('Cell Boundary')
    plt.show()
    
    
###########################################################################################################################
def overlay_nuclei_centers(image_data, nuclei: list[Nucleus], labels, label_colors):
    num_nuclei = len(nuclei)
    dots_image = np.zeros_like(image_data)
    
    for i in range(num_nuclei):
        x = nuclei[i].center[1]
        y= nuclei[i].center[0]
        color = label_colors[labels[i]] if labels[i] < len(label_colors) else (0, 0, 0)  # Default color is black for unknown labels
        
        cv2.circle(dots_image, (int(x), int(y)), 10, color, -1)  # Increase dot size

        # Overlay the dots image onto the original glomeruli image with increased opacity
        result_image = cv2.addWeighted(image_data, 0.35, dots_image, 1, 0)
        
    cv2.imshow('Nuclei Centers Overlay', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    return result_image


def generate_random_labels(codex_path, nuclei):
    
    with open(codex_path, 'r') as file:
        codex_names = file.readlines()
        codex_names = [name.strip().split(maxsplit=1)[-1] for name in codex_names]
        
        color_palette = sns.color_palette("colorblind", len(codex_names))  # Generate colorblind-friendly palette with seaborn
        rgb_values = [(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in color_palette]  # Convert seaborn color palette to RGB values
        
        codex_channels = np.linspace(0, len(nuclei)-1,dtype=int)
        labels = []
        
        for index in nuclei:
            cell_label = rd.randint(0, len(codex_names)-1)
            labels.append(cell_label)
    
    return labels, rgb_values, codex_names


# Wrapper function for XML generation
def make_xml_annotations(cell_names, nuclei: list[Nucleus], labels, filename_xml='XML_Annotation.xml',filename_json='JSON_Annotation.json', m=-1):
    annotations = wak.Annotation()
    annotations.add_names(cell_names)
    num_nuclei = len(nuclei)
    radius = 5

    for i in range(num_nuclei):
        x = nuclei[i].center[1]
        y= nuclei[i].center[0]
        point = Point(y, x)
        circle = point.buffer(5).simplify(tolerance=0.05, preserve_topology=False)
        annotations.add_shape(poly=circle, box_crs=[0, 0], structure=cell_names[labels[i]-1], name="") #update crs if they're relative to smaller mask to be top left corner
    print(annotations)
    if m != -1:
        filename_xml = filename_xml[0:-4] + str(m) + '.xml'
        filename_json = filename_json[0:-5] + str(m) + '.json'
    print(filename_xml, filename_json)
    annotations.xml_save(filename_xml)
    annotations.json_save(filename_json)
    return None


if __name__ == '__main__':
    file_path = 'data/Section6_CODEX.tif'
    image_array = load_tiff_as_array(file_path)

    print(f'Image array is {"x".join(map(str, image_array.shape))}')
    channel_indices = np.linspace(0,40,dtype=int)
    display_channel_heatmaps(image_array, channel_indices)

