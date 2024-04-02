import numpy as np
from logging import getLogger
from scipy.stats import skew, kurtosis

logger = getLogger()


def mask_array(pixel_values: np.ndarray) -> np.ma.MaskedArray:
    height, width = pixel_values.shape
    radius = min(height, width) // 2
    x, y = np.ogrid[:height, :width]

    center_x = width // 2
    center_y = height // 2

    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) > radius ** 2

    masked_array = np.ma.masked_where(mask, pixel_values)

    return masked_array


def valid_values(masked_array: np.ma.MaskedArray) -> np.ndarray:
    return masked_array[masked_array.mask == False]


def generate_feature_extractors() -> list[callable]:
    methods = [_channelwise_mean,
               _channelwise_stdev,
               _channelwise_median,
               _channelwise_kurtosis,
               _channelwise_skewness,
               _max_intensity,
               _min_intensity]

    logger.debug(f'Feature extraction methods: {methods}')
    return methods


def _channelwise_mean(pixel_values: np.ma.MaskedArray) -> np.ndarray:
    return np.ma.mean(pixel_values, axis=(0, 1))


def _channelwise_stdev(pixel_values: np.ma.MaskedArray) -> np.ndarray:
    return np.ma.std(pixel_values, axis=(0, 1))


def _channelwise_median(pixel_values: np.ma.MaskedArray) -> np.ndarray:
    return np.ma.median(pixel_values, axis=(0, 1))


def _channelwise_skewness(pixel_values: np.ma.MaskedArray) -> np.ndarray:
    channels_skewness = []
    for channel_image in pixel_values:
        channel_values = valid_values(channel_image)
        channel_skew = skew(channel_values.flatten())
        channels_skewness.append(channel_skew)
    return np.array(channels_skewness)


def _channelwise_kurtosis(pixel_values: np.ma.MaskedArray) -> np.ndarray:
    channels_kurtosis = []
    for channel_image in pixel_values:
        channel_values = valid_values(channel_image)
        channel_skew = kurtosis(channel_values.flatten())
        channels_kurtosis.append(channel_skew)
    return np.array(channels_kurtosis)


def _min_intensity(pixel_values: np.ma.MaskedArray) -> np.ndarray:
    return np.ma.min(pixel_values, axis=(0, 1))


def _max_intensity(pixel_values: np.ma.MaskedArray) -> np.ndarray:
    return np.ma.max(pixel_values, axis=(0, 1))

<<<<<<< HEAD
=======
    Returns: channel-wise mean of pixel values
    """
    return np.mean(pixel_values, axis=(0, 1))

def get_nuclei_size(nuclei_mask):
    pixel_cell_count = np.sum(nuclei_mask[:,:])
    return pixel_cell_count
>>>>>>> origin/main
