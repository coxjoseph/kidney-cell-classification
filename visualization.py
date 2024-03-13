from os import PathLike
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import tifffile


def load_tiff_as_array(path: Union[str, PathLike]) -> np.ndarray:
    """
    Load a TIFF image and convert it to a numpy array.

    Parameters:
    - file_path: Path to the TIFF file.

    Returns:
    - A numpy array representing the image.
    """
    with tifffile.TiffFile(path) as tiff:
        array = tiff.asarray()
    return array


def display_channel_heatmaps(array: np.ndarray, indices: list) -> None:
    """
    Display specified channels of the image as heatmaps.

    Parameters:
    - array: A numpy array representing the image.
    - indices: A list of indices of the channels to be displayed.
    """
    for index in indices:
        plt.figure(figsize=(10, 8))
        plt.imshow(array[:, :, index], cmap='hot')
        plt.title(f'Channel {index} Heatmap')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    file_path = 'data/codex.tif'
    image_array = load_tiff_as_array(file_path)

    print(f'Image array is {"x".join(map(str, image_array.shape))}')
    channel_indices = [0, 1, 2]
    display_channel_heatmaps(image_array, channel_indices)
