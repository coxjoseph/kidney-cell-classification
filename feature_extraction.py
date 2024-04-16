import numpy as np
from logging import getLogger

logger = getLogger()


def generate_feature_extractors() -> list[callable]:
    # To use this, add new features to the end of this methods variable. Don't call them :)
    #   Each feature extractor should expect an input array of a square numpy matrix of an area around a cell called
    #   pixel_values. This matrix will contain all channels in the codex. It should output an iterable, as the outputs
    #   will be unwrapped with .extend()
    methods = [_example_feature_extractor]
    logger.debug(f'feature extraction methods: {methods}')
    return methods


def _example_feature_extractor(pixel_values: np.ndarray) -> np.ndarray:
    """
    Example feature extractor, which implements the channel-wise mean of the pixel values.
    Args:
        pixel_values: numpy array of values near the cell for which features are supposed to be calculated.

    Returns: channel-wise mean of pixel values
    """
    return np.mean(pixel_values, axis=(0, 1))

# Calculate the total pixel area of a single nucleus in an already isolated window
def get_nuclei_size(nuclei_mask):
    pixel_cell_count = np.sum(nuclei_mask[:,:])
    return pixel_cell_count