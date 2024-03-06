import argparse
import logging
from images import load_images, generate_classified_image
from cells import segment_nuclei, calculate_radii_from_nuclei, create_cells
from feature_extraction import generate_feature_extractors
from clustering import cluster

logger = logging.getLogger()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to classify cells in a H&E stained image with a paired CODEX '
                                                 'file')
    parser.add_argument('--codex', '-c', required=True, help='Path to CODEX tiff file', type=str)
    parser.add_argument('--he', '-h', required=True, help='Path to H&E stained tiff file', type=str)
    parser.add_argument('--output', '-o', help='output tiff file (default to ./classified_stain.tif)',
                        type=str, default='classified_stain.tif')

    args_ = parser.parse_args()
    logger.debug(f'Received arguments: {args_}')
    return args_


if __name__ == '__main__':
    args = parse_arguments()

    brightfield, codex = load_images(args)

    nuclei = segment_nuclei(brightfield)
    radii = calculate_radii_from_nuclei(nuclei)
    cells = create_cells(nuclei, radii)

    feature_extractors = generate_feature_extractors()
    [cell.calculate_features(feature_extractors, codex) for cell in cells]
    cluster(cells)

    classified_image = generate_classified_image(brightfield, cells, args, save=True)
