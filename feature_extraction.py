import numpy as np
from logging import getLogger

logger = getLogger('classification')


def valid_values(masked_array: np.ma.MaskedArray) -> np.ndarray:
    return masked_array[masked_array.mask == False]


def     generate_feature_extractors() -> list[callable]:
    methods = [_channelwise_mean,
               _channelwise_stdev,
               _channelwise_median,
               _max_intensity,
               _min_intensity, 
               _radius]

    logger.debug(f'Feature extraction methods: {[method.__name__ for method in methods]}')
    return methods


def _channelwise_mean(pixel_values: np.ma.MaskedArray) -> np.ndarray:
    return np.ma.mean(pixel_values, axis=(1, 2))


def _channelwise_stdev(pixel_values: np.ma.MaskedArray) -> np.ndarray:
    return np.ma.std(pixel_values, axis=(1, 2))


def _channelwise_median(pixel_values: np.ma.MaskedArray) -> np.ndarray:
    return np.ma.median(pixel_values, axis=(1, 2))


def _min_intensity(pixel_values: np.ma.MaskedArray) -> np.ndarray:
    return np.ma.min(pixel_values, axis=(1, 2))


def _max_intensity(pixel_values: np.ma.MaskedArray) -> np.ndarray:
    return np.ma.max(pixel_values, axis=(1, 2))
  

def _radius(pixel_values):
    return [pixel_values.shape[0] // 2]


def get_nuclei_size(nuclei_mask):
    pixel_cell_count = np.sum(nuclei_mask[:,:])
    return pixel_cell_count


def calculate_cell_features(cell, feature_extractors: list[callable]):
    cell.calculate_features(feature_extractors)
    return cell
