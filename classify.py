import argparse
import logging

import numpy as np

from images import load_images, generate_classified_image, register_images
from cells import segment_nuclei_brightfield, segment_nuclei_dapi, calculate_radii_from_nuclei, create_cells, extract_nuclei_coordinates, create_nuclei, get_nucleus_mask_dapi, slice_nucleus_window, calculate_nuclei_sizes, remove_largest_nuclei
from feature_extraction import generate_feature_extractors, calculate_cell_features
from clustering import cluster
import sys
from concurrent.futures import ThreadPoolExecutor, wait


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to classify cells in a H&E stained image with a paired CODEX '
                                                 'file')
    parser.add_argument('--codex', '-c', required=True, help='Path to CODEX tiff file', type=str)
    parser.add_argument('--he', required=True, help='Path to H&E stained tiff file', type=str)
    parser.add_argument('--output', '-o', help='output tiff file (default to ./classified_stain.tif)',
                        type=str, default='classified_stain.tif')
    parser.add_argument('--dapi', '-d', help='DAPI layer of CODEX file (default to 0)', type=int, default=0)
    parser.add_argument('--njobs', '-j', help='Number of CPU cores to run in parallel', type=int, default=4)
    parser.add_argument('--verbose', '-v', help='Print debug output', action='store_true')

    args_ = parser.parse_args()
    return args_


def initialize_logger():
    format_str = '%(asctime)s [%(levelname)s]: %(message)s'
    logging.basicConfig(format=format_str, stream=sys.stdout)
    
    
if __name__ == '__main__':
    args = parse_arguments()
    initialize_logger()
    logger = logging.getLogger('classification')
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level)
    logger.debug(f'Received arguments: {args}')
    
    brightfield, codex = load_images(args, rotate_brightfield=True)
    
    nuclei_mask_dapi = segment_nuclei_dapi(codex, DAPI_index=args.dapi, visual_output=False)
    # nuclei_mask_brightfield = segment_nuclei_brightfield(brightfield, window_size=512, visual_output=False)
    nuclei_dapi = extract_nuclei_coordinates(nuclei_mask_dapi, downsample_factor=4, num_processes=args.njobs, visual_output=False)
    nuclei_dapi = calculate_nuclei_sizes(nuclei_dapi, nuclei_mask_dapi, window_size=128)
    # nuclei_brightfield = extract_nuclei_coordinates(nuclei_mask_brightfield, downsample_factor=4, num_processes=args.njobs, visual_output=False)
    # nuclei_mask_brightfield = remove_largest_nuclei(nuclei_brightfield, nuclei_mask_brightfield, cull_percent=0.05, visual_output=True)
    radii = calculate_radii_from_nuclei(nuclei_dapi, nuclei_mask_dapi, window_size=128)
    cells = create_cells(nuclei_dapi, radii)
    
    #registered_image = register_images(nuclei_mask_dapi, nuclei_mask_brightfield, max_features=100000, visual_output=True)
    
    # START OF TEST CODE
    #nuclei_subsample = [(1718,5018),(1986,1410),(4062,3084)] # These coordinates were grabbed in a prior run
    #nucleus_subsample = [nuclei[5000].center, nuclei[10000].center, nuclei[15000].center, nuclei[20000].center, nuclei[25000].center]
    #radii_subsample = [radii[5000], radii[10000], radii[15000], radii[20000], radii[25000]]
    #for n in range(5):
    #    nucleus_mask_local = slice_nucleus_window(nuclei_mask_dapi, nucleus_subsample[n], window_size=256)
    #    overlay_cell_boundaries(nucleus_mask_local, radii_subsample[n])
    #   overlay_nuclei_boundaries(nucleus_subsample[n], nuclei_mask_dapi, codex, DAPI_index=args.dapi)
    # END OF TEST CODE

    feature_extractors = generate_feature_extractors()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for cell in cells:
            futures.append(executor.submit(calculate_cell_features,
                                           cell=cell, feature_extractors=feature_extractors, codex=codex))

        wait(futures)
        cells = [processed_cell.result() for processed_cell in futures]

    cluster(cells)

    channel_last_codex = np.transpose(codex, (1, 2, 0))
    generate_classified_image(channel_last_codex[:, :, 0], cells, args, save=True)
