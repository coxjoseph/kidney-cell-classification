import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from images import load_images, generate_classified_image, register_images
from cells import segment_nuclei_brightfield, segment_nuclei_dapi, calculate_radii_from_nuclei, create_cells, extract_nuclei_coordinates, create_nuclei, get_nucleus_mask_dapi, slice_nucleus_window
from visualization import overlay_cell_boundaries, overlay_nuclei_boundaries
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
    parser.add_argument('--dapi', '-d', help='DAPI layer of CODEX file (default to 0)', type=int, default=0)
    parser.add_argument('--njobs', '-j', help='Number of CPU cores to run in parallel', type=int, default=4)

    args_ = parser.parse_args()
    logger.debug(f'Received arguments: {args_}')
    return args_


if __name__ == '__main__':
    args = parse_arguments()

    brightfield, codex = load_images(args)
    
    nuclei_mask_dapi = segment_nuclei_dapi(codex, DAPI_index=args.dapi, visual_output=False)
    nuclei_mask_brightfield = segment_nuclei_brightfield(brightfield, window_size=512, visual_output=False)
    nuclei = extract_nuclei_coordinates(nuclei_mask_dapi, downsample_factor=4, num_processes=args.njobs, visual_output=False)
    #nuclei = extract_nuclei_coordinates(nuclei_mask_brightfield, downsample_factor=4, num_processes=args.njobs, visual_output=False)
    radii = calculate_radii_from_nuclei(nuclei, nuclei_mask_dapi, window_size=128)
    cells = create_cells(nuclei, radii)
    
    # START OF TEST CODE
    #overlay_nuclei_boundaries(nuclei[0].center, codex, args.dapi)
    #nuclei_subsample = [(1718,5018),(1986,1410),(4062,3084)] # These coordinates were grabbed in a prior run
    #nucleus_subsample = [nuclei[5000].center, nuclei[10000].center, nuclei[15000].center, nuclei[20000].center, nuclei[25000].center]
    #radii_subsample = [radii[5000], radii[10000], radii[15000], radii[20000], radii[25000]]
    #for n in range(5):
    #    nucleus_mask_local = slice_nucleus_window(nuclei_mask_dapi, nucleus_subsample[n], window_size=256)
    #    overlay_cell_boundaries(nucleus_mask_local, radii_subsample[n])
    #    overlay_nuclei_boundaries(nucleus_subsample[n], nuclei_mask_dapi, codex, DAPI_index=args.dapi)
    # END OF TEST CODE

    #feature_extractors = generate_feature_extractors()
    #[cell.calculate_features(feature_extractors, codex) for cell in cells]
    #cluster(cells)

    #classified_image = generate_classified_image(brightfield, cells, args, save=True)
