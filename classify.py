import argparse
import logging
from images import load_images, generate_classified_image
from cells import segment_nuclei_brightfield, segment_nuclei_dapi, calculate_radii_from_nuclei, create_cells
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

    args_ = parser.parse_args()
    logger.debug(f'Received arguments: {args_}')
    return args_


if __name__ == '__main__':
    args = parse_arguments()

    brightfield, codex = load_images(args)
    
    # START OF TESTING CODE
    nuclei = [(5906, 7187)]; # Hardcoded nuclei value until nuclei extractor code is done
    radii = [15]; # Arbitrary for now
    cells = create_cells(nuclei, radii)
    cells[0].get_cell_mask_irregular(codex, DAPI_index=0, cyto_index=1, visual_output=True); # Cyto index is a placeholder, not currently used
    
    mask = segment_nuclei_dapi(codex, DAPI_index=0, visual_output=True)
    # END OF TESTING CODE
    
    
    # Temporarily commented out for testing
    #nuclei = segment_nuclei_brightfield(brightfield)
    #radii = calculate_radii_from_nuclei(nuclei)
    #cells = create_cells(nuclei, radii)

    #feature_extractors = generate_feature_extractors()
    #[cell.calculate_features(feature_extractors, codex) for cell in cells]
    #cluster(cells)

    #classified_image = generate_classified_image(brightfield, cells, args, save=True)
