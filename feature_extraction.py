import numpy as np
from logging import getLogger
from scipy.stats import skew

logger = getLogger()


def generate_feature_extractors() -> list[callable]:
    # To use this, add new features to the end of this methods variable. Don't call them :)
    #   Each feature extractor should expect an input array of a square numpy matrix of an area around a cell called
    #   pixel_values. This matrix will contain all channels in the codex. It should output an iterable, as the outputs
    #   will be unwrapped with .extend()
    methods = [_channel_mean,
               _area,
               _standard_dev,
               _max_intensity,
               _radius,
               _intensity_skewness,
               _minimum_intensity,
               _channel_moment,
               _correlation_coefficients]
    logger.debug(f'feature extraction methods: {methods}')
    return methods


def _channel_mean(pixel_values: np.ndarray) -> np.ndarray:
    """
   Feature extractor:  channel-wise mean of the pixel values.
    Args:
        pixel_values: numpy array of values near the cell for which features are supposed to be calculated.

    Returns: channel-wise mean of pixel values
    """
    return np.mean(pixel_values, axis=(0, 1))


def _area(pixel_values: np.ndarray) -> list[int]:
    # Assumes square array
    radius = pixel_values.shape[0] / 2
    return [np.pi * (radius ** 2)]


def _standard_dev(pixel_values: np.ndarray) -> np.ndarray:
    return np.std(pixel_values, axis=(0, 1))


def _max_intensity(pixel_values: np.ndarray) -> np.ndarray:
    return np.max(pixel_values, axis=(0, 1))


def _radius(pixel_values: np.ndarray) -> list[int]:
    return pixel_values.shape[0] / 2


def _intensity_skewness(pixel_values: np.ndarray) -> list[float]:
    x, y, = np.ogrid[:pixel_values.shape[0], :pixel_values.shape[1]]
    center_x, center_y = pixel_values.shape[0] // 2, pixel_values.shape[1] // 2
    radius = _radius(pixel_values)[0]  # Recall this is a list
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    mask = dist <= radius
    mask = mask.astype(bool)  # Does this do anything?

    channel_skewness = []
    for channel in range(pixel_values.shape[2]):
        channel_data = pixel_values[:, :, channel][mask]
        skewness = skew(channel_data)
        channel_skewness.append(float(skewness))  # Potential error here if skewness is angry
    return channel_skewness


def _minimum_intensity(pixel_values: np.ndarray) -> np.ndarray:
    return np.min(pixel_values, axis=(0, 1))


def _channel_moment(pixel_values: np.ndarray) -> list[float]:
    # TODO: this can likely be vectorized somehow
    moments = []
    for index in range(pixel_values.shape[-1]):
        channel = pixel_values[:, :, index]
        x, y = np.indices(channel.shape)

        zeroth_moment = channel.sum()
        if zeroth_moment != 0:
            m10 = (x * channel).sum()
            m01 = (y * channel).sum()
            first_moment_x = m10 / zeroth_moment
            first_moment_y = m01 / zeroth_moment
        else:
            first_moment_x, first_moment_y = 0, 0

        moments.extend([zeroth_moment, first_moment_x, first_moment_y])

    return moments


def _correlation_coefficients(pixel_vales: np.ndarray) -> np.ndarray:
    # Could be biased because of 0 values?
    # Guaranteed to produce some meaningless features because corr will always be 1 diagonal in every case.
    flattened_channels = [pixel_vales[:, :, i].flatten() for i in range(pixel_vales.shape[-1])]
    corr_matrix = np.corrcoef(flattened_channels)
    return corr_matrix.flatten()

