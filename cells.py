from dataclasses import dataclass
from typing import Union
import numpy as np
from logging import getLogger
from skimage.color import separate_stains, hdx_from_rgb, rgb_from_hdx
from skimage import filters, morphology, measure
import matplotlib.pyplot as plt

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

        logger.debug(f'{output_array.shape=}')
        logger.info('Channel-wise pixel values separated near nuclei...')
        return output_array

    def calculate_features(self, feature_extractors: list[callable], codex: np.ndarray) -> None:
        pixel_values = self.get_pixel_values(codex)  # THESE WILL BE SQUARE ARRAYS
        features = []
        [features.extend(feature_extractor(pixel_values)) for feature_extractor in feature_extractors]
        logger.info('Created features...')
        logger.debug(f'{features}')
        self.features = tuple(features)


def segment_nuclei(brightfield: np.ndarray, display: bool = True) -> list[Nucleus]:
    ihc_hdx = separate_stains(brightfield, hdx_from_rgb)

    if display:
        show_separated_stains(ihc_hdx)

    centers = get_center_points(brightfield, display=display)
    logger.info(f'Segmented {len(centers)} nuclei')
    return [Nucleus(center=center) for center in centers]


def calculate_radii_from_nuclei(nuclei: list[Nucleus]) -> tuple[int]:
    """
    Calculate a tuple of nucleus radii from a list of nuclei. The order of elements in the returned tuple corresponds to
        the order of the nuclei. This method assumes that all nuclei are spherical and only calculates the radius as the
        minimum distance between nuclei centers. For more advanced methods can use watershed or voronoi tessellation,
        both of which are common in histology.

    TODO: overload this method and implement Watershed?

    TODO: this method can be made to run in O(n log(n)) time by using trees (like KD trees).

    Args:
        nuclei: list of Nucleus objects to calculate the nuclei.

    Returns: tuple of integers, each element being the radius of the respective nucleus in the input list.
    """
    centroids = np.array([nucleus.center for nucleus in nuclei])

    distance = np.sqrt(((centroids[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)  # Replace self-distance with infinity

    min_distances = np.min(distance, axis=1)

    radii = min_distances / 2
    radii = tuple(map(int, radii))

    logger.info('Calculated radii...')
    logger.debug(f'{radii=}')
    return radii


def create_cells(nuclei: list[Nucleus], radii: tuple[int]) -> list[Cell]:
    if not len(nuclei) == len(radii):
        raise ValueError('Cannot initialize Cell objects: Radii and nuclei have different lengths: '
                         f'({len(nuclei)=} | {len(radii)=}).')

    return [Cell(nucleus=nucleus, radius=radius) for nucleus, radius in zip(nuclei, radii)]


def show_separated_stains(ihc_hdx: np.ndarray) -> None:
    hematoxylin_channel = ihc_hdx[:, :, 0]
    eosin_channel = ihc_hdx[:, :, 1]

    fix, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(rgb_from_hdx(ihc_hdx))
    ax[0].set_title("Original")
    ax[1].imshow(hematoxylin_channel, cmap='gray')
    ax[1].set_title("Hematoxylin Channel")
    ax[2].imshow(eosin_channel, cmap='gray')
    ax[2].set_title("Eosin Channel")

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()


def get_center_points(hematoxylin_channel: np.ndarray, display: bool = True) -> list[tuple]:
    threshold = filters.threshold_otsu(hematoxylin_channel)
    logger.debug(f'{threshold=}')
    binary = hematoxylin_channel > threshold

    cleaned = morphology.opening(binary, morphology.disk(3))
    cleaned = morphology.closing(cleaned, morphology.disk(3))

    labels = measure.label(cleaned)
    props = measure.regionprops(labels)

    centroids = [prop.centroid for prop in props]

    # centroids are in y, x, not x, y.
    centroids = [(centroid[1], centroid[0]) for centroid in centroids]

    if display:
        fix, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(hematoxylin_channel, cmap='gray')

        for centroid in centroids:
            ax.plot(centroid[0], centroid[1], 'r.')

        ax.set_title('Calculated centroids')
        ax.axis('off')
        plt.show()

    return centroids
