import numpy as np
from logging import getLogger
from scipy.stats import skew, kurtosis

logger = getLogger()


def valid_values(masked_array: np.ma.MaskedArray) -> np.ndarray:
    return masked_array[masked_array.mask == False]


def generate_feature_extractors() -> list[callable]:
    methods = [_channelwise_mean,
               _channelwise_stdev,
               _channelwise_median,
               # _channelwise_kurtosis,
               # _channelwise_skewness,
               _max_intensity,
               _min_intensity]

    logger.debug(f'Feature extraction methods: {[method.__name__ for method in methods]}')
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


def _radius(pixel_values):
    return pixel_values.shape[0] // 2
