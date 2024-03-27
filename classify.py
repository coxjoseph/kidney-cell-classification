import argparse
import logging
import numpy as np
from images import load_images, generate_classified_image
from cells import segment_nuclei_brightfield, segment_nuclei_dapi, calculate_radii_from_nuclei, create_cells, extract_nuclei_coordinates
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
    #parser.add_argument('--processcount', '-t', help='Number of CPU cores to run in parallel',
    #                    type=str, default='classified_stain.tif')

    parsed_args = parser.parse_args()
    logger.debug(f'Received arguments: {parsed_args}')
    return parsed_args


if __name__ == '__main__':
    args = parse_arguments()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - $(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - $(levelname)s - %(message)s')

    brightfield, codex = load_images(args)
    
    # START OF TESTING CODE
    #nuclei = [(5906, 7187)] # Hardcoded nuclei value until nuclei extractor code is done
    #nuclei =[(4062, 3086)]
    #radii = [15]; # Arbitrary for now
    #cells = create_cells(nuclei, radii)
    #nuclei_mask = cells[0].get_cell_mask_irregular(codex, DAPI_index=0, cyto_index=1, visual_output=True) # Cyto index is a placeholder, not currently used
    
    use_debug_coordinates = True
    if (not use_debug_coordinates):
        print('Beginning nuclei mask generation...', flush=True)
        nuclei_mask = segment_nuclei_dapi(codex, dapi_index=0, visual_output=False)
        nuclei_list = extract_nuclei_coordinates(nuclei_mask, downsample_factor=4, num_processes=8, visual_output=False)
        nuclei_subsample = [nuclei_list[10000], nuclei_list[20000], nuclei_list[30000]]  # Grab a small randomish sample of nuclei for testing
        radii = [1, 1, 1]  # Temporary values for debugging
        cells = create_cells(nuclei_subsample, radii)
        nuclei_mask = cells[0].get_cell_mask_irregular(codex, DAPI_index=0, cyto_index=1, visual_output=True)  # Cyto index is a placeholder, not currently used
    else:
        nuclei_subsample = [(1718,5018),(1986,1410),(4062,3084)] # These coordinates were grabbed with a ds factor=4
        radii = [1, 1, 1] # Temporary values for debugging
        cells = create_cells(nuclei_subsample, radii)
        nuclei_mask = nuclei_mask = cells[0].get_cell_mask_irregular(codex, DAPI_index=0, cyto_index=1, visual_output=True)  # Cyto index is a placeholder, not currently used
        nuclei_mask = nuclei_mask = cells[1].get_cell_mask_irregular(codex, DAPI_index=0, cyto_index=1, visual_output=True)  # Cyto index is a placeholder, not currently used
        nuclei_mask = nuclei_mask = cells[2].get_cell_mask_irregular(codex, DAPI_index=0, cyto_index=1, visual_output=True)  # Cyto index is a placeholder, not currently used
    
    # END OF TESTING CODE
    
    # Temporarily commented out for testing
    #nuclei = segment_nuclei_brightfield(brightfield)
    #radii = calculate_radii_from_nuclei(nuclei)
    #cells = create_cells(nuclei, radii)

    #feature_extractors = generate_feature_extractors()
    #[cell.calculate_features(feature_extractors, codex) for cell in cells]
    #cluster(cells)

    generate_classified_image(brightfield, cells, args, save=True)
