import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from typing import Union
from logging import getLogger
import umap

from cells import Cell

logger = getLogger()


def dimensionality_reduction(features: np.ndarray, num_components: Union[int, None] = None,
                             reduction_type: str = 'UMAP') -> np.ndarray:
    if reduction_type == 'UMAP':
        reducer = umap.UMAP(random_state=2024)
        reduced_features = reducer.fit_transform(features)
    else:
        pca = PCA(n_components=num_components)
        reduced_features = pca.fit_transform(features)
    logger.info('Reduced dimensionality of feature space...')
    return reduced_features


def cluster(cells: list[Cell], num_components: Union[int, None] = None,
            cluster_alg: str = 'KMeans', **kwargs) -> None:
    features = []
    for cell in cells:
        features.append(cell.features)

    features = np.array(features)
    logger.debug(f'{features.shape=}')
    reduced_features = dimensionality_reduction(features, num_components=num_components)
    logger.debug(f'{reduced_features.shape=}')

    if cluster_alg == 'dbscan':
        dbscan = DBSCAN(**kwargs)
        predictions = dbscan.fit_predict(reduced_features)
    else:
        kmeans = KMeans(**kwargs)
        predictions = kmeans.fit_predict(reduced_features)
    logger.info('Predicted each cell...')
    logger.debug(f'{predictions.shape=}')
    for i, cell in enumerate(cells):
        cell.label = predictions[i]
    logger.info('Cells clustered successfully!')
