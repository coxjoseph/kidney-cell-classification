import argparse
import tifffile
from logging import getLogger
import numpy as np
from skimage.transform import resize
from cells import Cell, slice_nucleus_window
import matplotlib.pyplot as plt
import cv2
import imutils

logger = getLogger()


def load_images(args_: argparse.Namespace, rotate_brightfield) -> tuple[np.ndarray, np.ndarray]:
    print('Loading images...', flush=True)
    codex_array, he_array = tifffile.TiffFile(args_.codex).asarray(), tifffile.TiffFile(args_.he).asarray()

    logger.debug(f'{codex_array.shape=} | {he_array.shape=}')
    
    # Rotate the brightfield to match the orientation of the CODEX
    print('Rotating brightfield...', flush=True)
    if (rotate_brightfield):
        # Coordinate space of the rotated brightfield
        num_rows = he_array.shape[1]
        num_columns = he_array.shape[0]
        
        brightfield_rotated = np.empty((num_rows, num_columns, 3), dtype=np.uint8)
        for n in range(0, num_columns):
            for m in range(0, num_rows):
                brightfield_rotated[m,n,:] = he_array[n, num_rows-m-1, :]
        he_array = brightfield_rotated        
       
    target_shape = he_array.shape
    #codex_array = resize(codex_array, output_shape=(target_shape[0], target_shape[1]), order=1, mode='reflect',
    #                     anti_aliasing=True)

    logger.info('Successfully loaded and resized images')
    logger.debug(f'f{codex_array.shape=} | {he_array.shape}')
    return he_array, codex_array

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
    keep_percent = 0.0001
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
    # TODO: can we just do this with the bf image? unsure.
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