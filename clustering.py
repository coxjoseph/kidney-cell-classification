import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from typing import Union
from logging import getLogger

from cells import Cell

logger = getLogger()


def dimensionality_reduction(features: np.ndarray, num_components: Union[int, None] = None):
    pca = PCA(n_components=num_components)
    reduced_features = pca.fit_transform(features)  # reduce dimensionality
    logger.info('Reduced dimensionality of feature space...')
    return reduced_features


def cluster(cells: list[Cell], num_components: Union[int, None] = None, **kwargs) -> None:
    # TODO: can we refactor this to make it faster? Is this even a slowdown? Need to profile.
    features = []
    for cell in cells:
        features.append(cell.features)

    features = np.array(features)
    logger.debug(f'{features.shape=}')
    reduced_features = dimensionality_reduction(features, num_components=num_components)
    logger.debug(f'{reduced_features.shape=}')

    # TODO: We dont have to use dbscan if we dont want to :)
    dbscan = DBSCAN(**kwargs)
    predictions = dbscan.fit_predict(reduced_features)
    logger.info('Predicted each cell...')
    logger.debug(f'{predictions.shape=}')
    for i, cell in enumerate(cells):
        cell.label = predictions[i]
    logger.info('Cells clustered successfully!')
