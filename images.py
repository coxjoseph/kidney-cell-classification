import argparse
import tifffile
from logging import getLogger
import numpy as np
from skimage.transform import resize
from cells import Cell
import matplotlib.pyplot as plt
from functools import partial

logger = getLogger()


def mapping_function(*args, **kwargs) -> callable:
    # ToDo: write an accurate mapping function from bf to codex.
    return partial(resize,  *args, **kwargs)


def load_images(args_: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    codex_array, he_array = tifffile.TiffFile(args_.codex).asarray(), tifffile.TiffFile(args_.he).asarray()

    logger.debug(f'{codex_array.shape=} | {he_array.shape=}')

    target_shape = he_array.shape
    mapper = mapping_function(output_shape=(target_shape[0], target_shape[1]), order=1, mode='reflect', anti_aliasing=True)
    he_array = mapper(codex_array)

    logger.info('Successfully loaded and resized images')
    logger.debug(f'f{codex_array.shape=} | {he_array.shape}')
    return he_array, codex_array


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
