from dataclasses import dataclass
from typing import Union
import numpy as np
from logging import getLogger


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


def segment_nuclei(brightfield: np.ndarray) -> list[Nucleus]:
    # Example for now:
    example_nuclei = [Nucleus(center=(x, y)) for x, y in zip(range(5), range(5))]
    logger.info(f'Segmented {len(example_nuclei)} nuclei')
    # TODO: create a method/methods that look at the brightfield array (should we switch to the codex image?) and
    #  create a list of center-points for nuclei. Once we have that uncomment below:

    # centers = get_center_points(brightfield)
    # logger.info(f'Segmented {len(centers)} nuclei')
    # return [Nucleus(center=center) for center in centers]
    return example_nuclei


def calculate_radii_from_nuclei(nuclei: list[Nucleus], brightfield: Union[np.ndarray, None] = None) -> list[int]:
    # Example for now:
    example_radii = [1] * len(nuclei)

    # TODO: look at the nuclei and, optionally, the brightfield image (again, maybe codex?) and creates a list of
    #  radii for each nucleus. These lists should be linked to each other (best to make them as tuples to enforce
    #  this? maybe a dictionary? seems overkill). This could be density based, or could be just a flat value that we
    #  calculate based on the centers. I don't see a way to do this in less than O(n^2) time if we do something
    #  fancy, but we could speedup with just selecting the half the minimum distance between centers I guess. Once
    #  implemented uncomment below:

    # radii = get_radii(nuclei, brightfield)
    # logger.info('Calculated radii...')
    # logger.debug(f'DEBUG: {radii=}')
    # return radii
    return example_radii


def create_cells(nuclei: list[Nucleus], radii: list[int]) -> list[Cell]:
    if not len(nuclei) == len(radii):
        raise ValueError('Cannot initialize Cell objects: Radii and nuclei have different lengths: '
                         f'({len(nuclei)=} | {len(radii)=}).')

    return [Cell(nucleus=nucleus, radius=radius) for nucleus, radius in zip(nuclei, radii)]
