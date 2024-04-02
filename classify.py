import argparse
import logging
from images import load_images, generate_classified_image
from cells import segment_nuclei_dapi, calculate_radii_from_nuclei, create_cells, extract_nuclei_coordinates
from feature_extraction import generate_feature_extractors
from clustering import cluster
<<<<<<< HEAD
import sys
=======
from feature_extraction import get_nuclei_size

logger = logging.getLogger()
>>>>>>> origin/main


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to classify cells in a H&E stained image with a paired CODEX '
                                                 'file')
    parser.add_argument('--codex', '-c', required=True, help='Path to CODEX tiff file', type=str)
    parser.add_argument('--he', required=True, help='Path to H&E stained tiff file', type=str)
    parser.add_argument('--output', '-o', help='output tiff file (default to ./classified_stain.tif)',
                        type=str, default='classified_stain.tif')
    parser.add_argument('--dapi', '-d', help='DAPI layer of CODEX file (default to 0)', type=int, default='0')
    parser.add_argument('--threads', '-t', help='Number of CPU cores to run in parallel', type=int, default='8')
    parser.add_argument('--verbose', '-v', help='Print debug output', action='store_true')

    args_ = parser.parse_args()
    return args_


def initialize_logger(verbose: bool):
    log_level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s [%(levelname)s]: %(message)s'
    logging.basicConfig(level=log_level, format=format_str, stream=sys.stdout)


if __name__ == '__main__':
    args = parse_arguments()
    initialize_logger(args.verbose)
    logger = logging.getLogger()

    logger.debug(f'Received arguments: {args}')
    brightfield, codex = load_images(args)
    
<<<<<<< HEAD
    nuclei_mask = segment_nuclei_dapi(codex, dapi_index=args.dapi)
    nuclei = extract_nuclei_coordinates(nuclei_mask, downsample_factor=4, num_threads=args.threads, visual_output=False)
    radii = calculate_radii_from_nuclei(nuclei, codex, dapi_index=args.dapi, window_size=128)
=======
    # START OF TESTING CODE

    # nuclei_subsample = [(1718,5018),(1986,1410),(4062,3084)] # These coordinates were grabbed in a prior run
    # nuclei_mask = get_nucleus_mask(nuclei_subsample[2], codex, DAPI_index=args.dapi, visual_output=True)

    # END OF TESTING CODE
    
    nuclei_mask = segment_nuclei_dapi(codex, DAPI_index=args.dapi)
    nuclei = extract_nuclei_coordinates(nuclei_mask, downsample_factor=4, num_processes=args.njobs, visual_output=False)
    radii = calculate_radii_from_nuclei(nuclei, codex, DAPI_index=args.dapi, window_size=128)
>>>>>>> origin/main
    cells = create_cells(nuclei, radii)

    feature_extractors = generate_feature_extractors()
    [cell.calculate_features(feature_extractors, codex) for cell in cells]
    cluster(cells)

<<<<<<< HEAD
    generate_classified_image(brightfield, cells, args, save=True)
=======

    #classified_image = generate_classified_image(brightfield, cells, args, save=True)
>>>>>>> origin/main
