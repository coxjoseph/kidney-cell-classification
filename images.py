import argparse
from typing import Optional

import tifffile
from logging import getLogger
import numpy as np
from skimage.transform import resize
from cells import Cell
import matplotlib.pyplot as plt
import cv2
from functools import partial

logger = getLogger()


def downscale(image: np.ndarray, scaling_factor: Optional[int] = 3):
    if scaling_factor is None:
        scaling_factor = 3
    width = image.shape[0] // scaling_factor
    height = image.shape[1] // scaling_factor

    dim = (width, height)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def mapping_function(*args, **kwargs) -> callable:
    return partial(resize,  *args, **kwargs)


def load_images(args_: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    codex_array, he_array = tifffile.TiffFile(args_.codex).asarray(), tifffile.TiffFile(args_.he).asarray()
    # TODO: make a way we can choose whether to do this... and specify which dimensions to do it to
    codex_array = np.transpose(codex_array)

    logger.debug(f'{codex_array.shape=} | {he_array.shape=}')
    logger.debug(f'{codex_array.dtype=} | {he_array.dtype=}')

    downsampled_codex = downscale(codex_array, scaling_factor=args_.scaling_factor)
    downsampled_he = downscale(he_array, scaling_factor=args_.scaling_factor)
    del codex_array
    del he_array
    target_shape = downsampled_codex.shape

    mapper = mapping_function(output_shape=(target_shape[0], target_shape[1]), anti_aliasing=False)
    downsampled_he = mapper(downsampled_he)

    logger.info('Successfully loaded and resized images')
    logger.debug(f'{downsampled_codex.shape=} | {downsampled_he.shape=}')
    return downsampled_codex, downsampled_he


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
