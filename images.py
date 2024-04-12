import argparse
import tifffile
from logging import getLogger
import numpy as np
from skimage.transform import resize
from cells import Cell
import matplotlib.pyplot as plt
import cv2

logger = getLogger()


def load_images(args_: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    codex_array, he_array = tifffile.TiffFile(args_.codex).asarray(), tifffile.TiffFile(args_.he).asarray()

    logger.debug(f'{codex_array.shape=} | {he_array.shape=}')

    target_shape = he_array.shape
    #codex_array = resize(codex_array, output_shape=(target_shape[0], target_shape[1]), order=1, mode='reflect',
    #                     anti_aliasing=True)

    logger.info('Successfully loaded and resized images')
    logger.debug(f'f{codex_array.shape=} | {he_array.shape}')
    return he_array, codex_array

# Reference: https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
def register_images(dapi_mask: np.ndarray, brightfield_mask: np.ndarray) -> np.ndarray:
    """
    Keypoint-based image registration
    
    Parameters:
    - dapi_mask: A binary mask of DAPI nuclei segmentation from the CODEX.
    - brightfield_mask: A binary mask of nuclei segmentation from the brightfield H&E image. The dimensions and aspect 
                        ratio do not need to match the dapi_mask.
    """
    
    # Detect keypoints from the binary masks
    max_features = 500
    orb = cv2.ORB_create(max_features)
    (kpsA, descsA) = orb.detectAndCompute(dapi_mask, None) # Image to be warped
    (kpsB, descsB) = orb.detectAndCompute(brightfield_mask, None) # Template image
    
    # Perform feature matching
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
    
    # Sort the matches by their hamming distance
    matches = sorted(matches, key=lambda x:x.distance)
    keep_percent = 0.15
    keep = int(len(matches) * keepPercent) # Calculate the number of matches to keep
    matches = matches[:keep] # Discard the less favorable matches
    
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
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(dapi_mask, H, (w, h))
    
	# return the aligned image
    return aligned

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