import logging
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import threading
from math import sqrt
from dataclasses import dataclass
from typing import Union
from skimage.filters import threshold_isodata
from feature_extraction import mask_array

logger = logging.getLogger()


@dataclass
class Nucleus:
    center: tuple


@dataclass
class Cell:
    nucleus: Nucleus
    radius: int
    features: Union[tuple, None] = None
    label: Union[int, None] = None

    def get_pixel_values(self, codex: np.ndarray) -> np.ndarray:
        r = int(self.radius)

        center_x, center_y = self.nucleus.center
        start_x, end_x = max(center_x + r, 0), min(center_x + r, codex.shape[0])
        start_y, end_y = max(center_y - r, 0), min(center_y + r, codex.shape[1])

        output_array = np.zeros((2 * r, 2 * r, codex.shape[2]))

        x_len = abs(end_x - start_x) // 2
        y_len = abs(end_y - start_y) // 2
        x, y = np.ogrid[-x_len:x_len, -y_len:y_len]
        distance_squared = x ** 2 + y ** 2
        mask = distance_squared <= (r ** 2)

        # Copy the relevant parts of each channel in codex
        for i in range(codex.shape[2]):
            cropped_channel = codex[start_x:end_x, start_y:end_y, i]
            # Make sure the cropped_channel fits (I think this covers edge cases mentioned above)
            min_x, min_y = max(r - center_x, 0), max(r - center_y, 0)
            max_x, max_y = min_x + cropped_channel.shape[0], min_y + cropped_channel.shape[1]
            output_array[min_x:max_x, min_y:max_y, i] = cropped_channel
        output_array = mask_array(output_array)
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
    upper_m = int(nucleus_coordinates[0] - mask_size / 2)
    lower_m = int(nucleus_coordinates[0] + mask_size / 2)
    left_n = int(nucleus_coordinates[1] - mask_size / 2)
    right_n = int(nucleus_coordinates[1] + mask_size / 2)

    # If the bounding box goes outside the image, shift it such that it is inside the image
    if left_n < 0:
        right_n = right_n - left_n
        left_n = 0
    if right_n >= codex_shape[2]:
        left_n = left_n - (right_n - codex_shape[2]) - 1
        right_n = codex_shape[2] - 1
    if upper_m < 0:
        lower_m = lower_m - upper_m
        upper_m = 0
    if lower_m >= codex_shape[1]:
        upper_m = upper_m - (lower_m - codex_shape[1]) - 1
        lower_m = codex_shape[1] - 1
    return [upper_m, lower_m, left_n, right_n]


def get_nucleus_mask(nucleus_coordinates, codex: np.ndarray, dapi_index, mask_size=256, opening_radius=2.5,
                     isolated=True, visual_output=False) -> np.ndarray:
    # Get bounding box coordinates
    upper_m, lower_m, left_n, right_n = get_bounding_box(nucleus_coordinates, mask_size, codex_shape=codex.shape)

    # Threshold nucleus stain
    subset_indices = (dapi_index, slice(upper_m, lower_m), slice(left_n, right_n))
    nuclei_mask = codex[subset_indices]

    threshold_scaler = 1.1  # Manual adjustment factor for automatic threshold
    threshold = threshold_isodata(nuclei_mask) * threshold_scaler  # Automatically obtain a threshold value
    nuclei_mask = (nuclei_mask > threshold).astype(np.uint8)  # Binarizes the image

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


def process_subset(image: np.ndarray, nuclei_mask: np.ndarray, downsample_factor: int) -> list[Nucleus]:
    thread = threading.get_ident()
    nuclei_list = []

    for m in range(image.shape[0]):
        for n in range(0, nuclei_mask.shape[1]):
            if nuclei_mask[m][n]:
                nuc = Nucleus((m * downsample_factor + downsample_factor // 2,
                               n * downsample_factor + downsample_factor // 2))
                nuclei_list.append(nuc)

                cv2.floodFill(nuclei_mask, None, (n, m), 0)  # Works?
                if len(nuclei_list) % 500 == 0:
                    logger.debug(f'Thread {thread}: Current nucleus count {len(nuclei_list)}')

    logger.debug(f'Thread {thread} finished...')
    return nuclei_list


def process_image(image_subset: np.ndarray, nuclei_mask: np.ndarray, result_list: list,
                  thread_lock: threading.Lock,
                  downsample_factor: int = None):
    logger.debug(f'{image_subset.shape=}')
    nuc_list = process_subset(image_subset, nuclei_mask, downsample_factor)
    with thread_lock:
        result_list.extend(nuc_list)


def extract_nuclei_coordinates(nuclei_mask: np.ndarray,
                               downsample_factor: int = 2,
                               num_threads: int = 8, visual_output: bool = False) -> list[Nucleus]:

    scale_factor = 1 / downsample_factor
    downsampled_mask = cv2.resize(nuclei_mask, None,
                                  fx=scale_factor,
                                  fy=scale_factor,
                                  interpolation=cv2.INTER_AREA)

    if visual_output:
        plt.figure(figsize=(10, 8))
        plt.imshow(downsampled_mask, cmap='hot')
        plt.title(f'Downsampled mask')
        plt.colorbar()
        plt.show()

    logger.info('Beginning nuclei coordinate extraction...')

    height, width = nuclei_mask.shape[:2]
    part_height = height // num_threads
    image_parts = [nuclei_mask[i * part_height:(i + 1) * part_height, :] for i in range(num_threads)]

    results = []
    lock = threading.Lock()

    threads = []
    for image_part in image_parts:
        thread = threading.Thread(target=process_image, args=(image_part,
                                                              nuclei_mask,
                                                              results,
                                                              lock,
                                                              downsample_factor,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    logger.info(f'Nuclei coordinates extracted: found {len(results)} nuclei.')
    return results


def make_circular_kernel(kern_radius) -> np.ndarray:
    center = kern_radius
    x, y = np.ogrid[:2 * kern_radius, :2 * kern_radius]
    kernel = ((x - center) ** 2 + (y - center) ** 2 <= kern_radius ** 2).astype(np.uint8)
    return kernel


def segment_nuclei_brightfield(brightfield: np.ndarray) -> list[Nucleus]:
    example_nuclei = [Nucleus(center=(x, y)) for x, y in zip(range(5), range(5))]
    logger.info(f'Segmented {len(example_nuclei)} nuclei')
    # TODO: create a method/methods that look at the brightfield array (should we switch to the codex image?) and
    #  create a list of center-points for nuclei. Once we have that uncomment below:
    return example_nuclei


def segment_nuclei_dapi(codex: np.ndarray, dapi_index: int, erosion_radius: int = 2.5, visual_output: bool = False) \
        -> np.ndarray:
    nuclei_mask = codex[:, :, dapi_index]
    threshold = threshold_isodata(nuclei_mask)
    nuclei_mask = (nuclei_mask > threshold).astype(np.uint8)
    kernel = make_circular_kernel(erosion_radius)
    nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_ERODE, kernel)

    if visual_output:
        plt.figure(figsize=(10, 8))
        plt.imshow(nuclei_mask, cmap='hot', interpolation=None)
        plt.title(f'DAPI Nuclei Mask')
        plt.colorbar()
        plt.show()

    return nuclei_mask


def isolate_nuclei(nuclei_window: np.ndarray, opening_radius, visual_output=False) -> np.ndarray:
    isolated_window = nuclei_window.copy()
    primary_nuclei_marker = 2  # Arbitrary value to separate the center nuclei from other nuclei visible in the window
    center = int(nuclei_window.shape[0] / 2)
    cv2.floodFill(isolated_window, None, (center, center),
                  primary_nuclei_marker)  # Give all connected pixels the primary nuclei marker
    isolated_window = (isolated_window == primary_nuclei_marker).astype(np.uint8)

    kernel = make_circular_kernel(opening_radius)
    isolated_window = cv2.morphologyEx(isolated_window, cv2.MORPH_DILATE, kernel)

    if visual_output:
        plt.figure(figsize=(4, 4))
        plt.imshow(isolated_window, cmap='hot')
        plt.title(f'Isolated Nuclei')
        plt.colorbar()
        plt.show()

    return isolated_window


def calculate_radii_from_nuclei(nuclei, codex: np.ndarray, dapi_index: int, window_size=256) -> list[float]:
    radii = []
    logger.info('Calculating radii from nuclei list...')
    for nucleus in nuclei:
        nucleus_mask = get_nucleus_mask(nucleus.center, codex, dapi_index, mask_size=window_size, isolated=False)
        _, labels = cv2.connectedComponents(nucleus_mask.astype(np.uint8))
        num_nearby_nuclei = np.max(labels)
        if num_nearby_nuclei == 0:
            num_nearby_nuclei = 1
        average_cell_area = (window_size * window_size) / num_nearby_nuclei
        radius = sqrt(average_cell_area / math.pi)
        radii.append(radius)
    return radii


def create_cells(nuclei: list[Nucleus], radii: list[float]) -> list[Cell]:
    if not len(nuclei) == len(radii):
        raise ValueError('Cannot initialize Cell objects: Radii and nuclei have different lengths: '
                         f'({len(nuclei)=} | {len(radii)=}).')

    return [Cell(nucleus=nucleus, radius=int(radius)) for nucleus, radius in zip(nuclei, radii)]


def create_nuclei(nuclei_coordinates) -> list[Nucleus]:
    nuclei = []
    for nuc in nuclei_coordinates:
        nuclei.append(Nucleus(center=nuc))
    return nuclei
