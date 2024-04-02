import argparse
import logging
import numpy as np
from images import load_images, generate_classified_image
from cells import segment_nuclei_brightfield, segment_nuclei_dapi, calculate_radii_from_nuclei, create_cells, extract_nuclei_coordinates, create_nuclei, get_nucleus_mask
from visualization import overlay_cell_boundaries
from skimage import io, transform
from feature_extraction import generate_feature_extractors
from clustering import cluster

logger = logging.getLogger()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to classify cells in a H&E stained image with a paired CODEX '
                                                 'file')
    parser.add_argument('--codex', '-c', required=True, help='Path to CODEX tiff file', type=str)
    parser.add_argument('--he', required=True, help='Path to H&E stained tiff file', type=str)
    parser.add_argument('--output', '-o', help='output tiff file (default to ./classified_stain.tif)',
                        type=str, default='classified_stain.tif')
    
    # TODO: FINISH THESE
    #parser.add_argument('--dapi', '-d', help='DAPI layer of CODEX file (default to 0)',
    #                    type=str, default='0')
    #parser.add_argument('--cyto', '-d', help='Cytoplasm layer of CODEX file (default to ./classified_stain.tif)',
    #                    type=str, default='classified_stain.tif')
    #parser.add_argument('--processcount', '-t', help='Number of CPU cores to run in parallel', type=str, default='8')

    args_ = parser.parse_args()
    logger.debug(f'Received arguments: {args_}')
    return args_


if __name__ == '__main__':
    args = parse_arguments()

    brightfield, codex = load_images(args)
    DAPI_index = 0 # Remove once arg parser is done
    
    # START OF TESTING CODE
    #    nuclei_subsample = [(1718,5018),(1986,1410),(4062,3084)] # These coordinates were grabbed in a prior run
    #    nuclei_mask = get_nucleus_mask(nuclei_subsample[2], codex, DAPI_index=0, visual_output=True)
    #    for codex_index in range(0, 40):
    #        show_codex_window(cells[2], codex_index, codex)
    
    #nuclei_subsample = [nuclei_list[10000], nuclei_list[20000], nuclei_list[30000]]# Grab a small randomish sample of nuclei for testing
    #nuclei_subsample = [(1718,5018),(1986,1410),(4062,3084)] # These coordinates were grabbed in a prior run
    #nuclei_mask = get_nucleus_mask(nuclei_subsample[1], codex, DAPI_index=0, isolated=False, visual_output=False)
    #overlay_cell_boundaries(nuclei_mask, 18)
    # END OF TESTING CODE
    
    nuclei_mask = segment_nuclei_dapi(codex, DAPI_index=DAPI_index)
    nuclei = extract_nuclei_coordinates(nuclei_mask, downsample_factor=4, num_processes=8, visual_output=False)
    radii = calculate_radii_from_nuclei(nuclei, codex, DAPI_index=DAPI_index, window_size=128)
    cells = create_cells(nuclei, radii)
    
    sample_mask = get_nucleus_mask(nuclei[15000].center, codex, DAPI_index=0, isolated=False)
    sample_radius = radii[15000]
    overlay_cell_boundaries(sample_mask, sample_radius)

    #feature_extractors = generate_feature_extractors()
    #[cell.calculate_features(feature_extractors, codex) for cell in cells]
    #cluster(cells)

    #classified_image = generate_classified_image(brightfield, cells, args, save=True)
