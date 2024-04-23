import argparse
from typing import Optional, Tuple, Callable, Union

import tifffile
from logging import getLogger
import numpy as np
from skimage.transform import resize
from cells import Cell, slice_nucleus_window
import matplotlib.pyplot as plt
import cv2
from functools import partial
import imutils

logger = getLogger()



def load_images(args_: argparse.Namespace, rotate_brightfield) -> tuple[np.ndarray, np.ndarray]:
    print('Loading images...', flush=True)
    codex_array, he_array = tifffile.TiffFile(args_.codex).asarray(), tifffile.TiffFile(args_.he).asarray()

    logger.info('Loaded CODEX and Brighfield tiff files...')
    logger.debug(f'{codex_array.shape=} | {he_array.shape=}')
    
    # Rotate the brightfield to match the orientation of the CODEX
    if (rotate_brightfield):
        print('Rotating brightfield...', flush=True)
        # Transpose the array to perform a 90-degree counterclockwise rotation
        he_array = np.transpose(he_array, axes=(1, 0, 2))  # Swap rows and columns
    
        # Flip the array horizontally to complete the rotation
        he_array = np.flip(he_array, axis=0)  # Flip along the first axis (rows)

    downsampled_codex, downsampled_he = preprocessing(codex_array, he_array)
    logger.info('Successfully loaded and resized images')
    logger.debug(f'{downsampled_codex.shape=} | {downsampled_he.shape=}')
    return downsampled_codex, downsampled_he

# Reference: https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
def register_images(dapi_mask: np.ndarray, brightfield_mask: np.ndarray, max_features, visual_output=False) -> np.ndarray:
    """
    Keypoint-based image registration
    
    Parameters:
    - dapi_mask: A binary mask of DAPI nuclei segmentation from the CODEX.
    - brightfield_mask: A binary mask of nuclei segmentation from the brightfield H&E image. The dimensions and aspect 
                        ratio do not need to match the dapi_mask.
    """
    print('Aligning images...', flush=True)
    
    # Take distance transform of the two nuclei masks
    #dapi_mask = cv2.distanceTransform(np.uint8(dapi_mask), cv2.DIST_L2, 3)
    #brightfield_mask = cv2.distanceTransform(np.uint8(brightfield_mask), cv2.DIST_L2, 3)
    
    # Convert the masks in uint8 arrays ranging from 0 to 255
    dapi_mask = dapi_mask.astype(np.uint8) * 255
    brightfield_mask = brightfield_mask.astype(np.uint8) * 255
    
    # Detect keypoints from the binary masks
    orb = cv2.ORB_create(max_features)
    (kpsA, descsA) = orb.detectAndCompute(dapi_mask, None) # Image to be warped
    (kpsB, descsB) = orb.detectAndCompute(brightfield_mask, None) # Template image
    
    # Perform feature matching
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
    
    # Sort the matches by their hamming distance
    matches = sorted(matches, key=lambda x:x.distance)
    keep_percent = 100/max_features
    keep = int(len(matches) * keep_percent) # Calculate the number of matches to keep
    matches = matches[:keep] # Discard the less favorable matches
    
    print(f'Matches: {len(matches)}')
    if visual_output:
        matchedVis = cv2.drawMatches(dapi_mask, kpsA, brightfield_mask, kpsB, matches, None, matchesThickness=150)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)
    
    # Coordinates of matches
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    for (i, m) in enumerate(matches):
        # Create mapping between the two coordinate spaces
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt
        

        
    # Calculate homography matrix
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    
	# Align the images using the homography matrix
    (h, w) = brightfield_mask.shape[:2]
    aligned_dapi = cv2.warpPerspective(dapi_mask, H, (w, h))
    
    if (visual_output):
        target_coordinates = (5000,5000) # Arbitrary location on the image to compare
        dapi_slice = slice_nucleus_window(aligned_dapi, target_coordinates, window_size=512)
        brightfield_slice = slice_nucleus_window(brightfield_mask, target_coordinates, window_size=512)
        
        # Show entire mask registration
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(aligned_dapi, cmap='gray')
        plt.title('Aligned DAPI Segmentation')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(brightfield_mask, cmap='gray')
        plt.title('Mask from H&E Segmentation')
        plt.axis('off')
        plt.show()
        
        # Show small region registration
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(dapi_slice, cmap='gray')
        plt.title('Mask from DAPI Segmentation')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(brightfield_slice, cmap='gray')
        plt.title('Mask from H&E Segmentation')
        plt.axis('off')
        plt.show()
    
	# return the aligned image
    return aligned_dapi

def generate_classified_image(brightfield: np.ndarray,
                              cells: list[Cell],
                              args: argparse.Namespace,
                              save: bool = True) -> None:
    centers = []
    labels = []

    for cell in cells:
        centers.append(cell.nucleus.center)
        labels.append(cell.label)

    num_colors = len(set(labels))
    if num_colors <= 10:
        cmap = plt.get_cmap('tab10')
    elif num_colors <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        raise ValueError(f'Number of labels  ({len(set(labels))}) is more than the number of colors we can generate. '
                         f'Implement more colors or change the clustering parameters to output fewer labels')

    plt.figure()
    plt.imshow(brightfield)
    for point, label in zip(centers, labels):
        x, y = point
        color = cmap(label)
        plt.scatter(x, y, color=color, s=100)  # size, can adjust if needed

    plt.axis('off')
    if save:
        plt.savefig(args.output)
    plt.show()
    logger.info(f'Saved image at {args.output}')
    
def global_dilate(nuclei_mask: np.ndarray, dilation_radius) -> np.ndarray:
    nuclei_mask_dilated = np.copy(nuclei_mask) # Preserve the original mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_radius, 2*dilation_radius))
    print('Beginning dilation...', flush=True)
    nuclei_mask_dilated = cv2.dilate(nuclei_mask, kernel)
    return nuclei_mask_dilated
