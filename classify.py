import argparse
import logging
import numpy as np
from images import load_images, generate_classified_image
from cells import segment_nuclei_brightfield, segment_nuclei_dapi, calculate_radii_from_nuclei, create_cells, extract_nuclei_coordinates, create_nuclei, get_nucleus_mask, Nucleus
from visualization import overlay_cell_boundaries, make_xml_annotations, overlay_nuclei_centers, generate_random_labels
from skimage import io, transform
from feature_extraction import generate_feature_extractors
from clustering import cluster
from feature_extraction import get_nuclei_size
import cv2

logger = logging.getLogger()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to classify cells in a H&E stained image with a paired CODEX '
                                                 'file')
    parser.add_argument('--codex', '-c', required=True, help='Path to CODEX tiff file', type=str)
    parser.add_argument('--he', required=True, help='Path to H&E stained tiff file', type=str)
    parser.add_argument('--output', '-o', help='output tiff file (default to ./classified_stain.tif)',
                        type=str, default='classified_stain.tif')
    parser.add_argument('--dapi', '-d', help='DAPI layer of CODEX file (default to 0)', type=int, default='0')
    parser.add_argument('--njobs', '-j', help='Number of CPU cores to run in parallel', type=int, default='8')

    args_ = parser.parse_args()
    logger.debug(f'Received arguments: {args_}')
    return args_


if __name__ == '__main__':
    args = parse_arguments()

    brightfield, codex = load_images(args)
    
    # START OF TESTING CODE

    # nuclei_subsample = [(1718,5018),(1986,1410),(4062,3084)] # These coordinates were grabbed in a prior run
    # nuclei_mask = get_nucleus_mask(nuclei_subsample[2], codex, DAPI_index=args.dapi, visual_output=True)

    # END OF TESTING CODE
    
    #nuclei_mask = segment_nuclei_dapi(codex, DAPI_index=args.dapi)
    nuclei_mask = segment_nuclei_brightfield(brightfield)
    nuclei = extract_nuclei_coordinates(nuclei_mask, downsample_factor=4, num_processes=args.njobs, visual_output=False)
    radii = calculate_radii_from_nuclei(nuclei, codex, DAPI_index=args.dapi, window_size=128)
    cells = create_cells(nuclei, radii)
    
    #print('hii')
    #print(type(nuclei))
    #print(cells.Nucleus[0])
    #rint(type(nuclei[0]))
    #print(nuclei[0])
    #print(nuclei)
    #print('>:)')

    #feature_extractors = generate_feature_extractors()
    #[cell.calculate_features(feature_extractors, codex) for cell in cells]
    #cluster(cells)


    #classified_image = generate_classified_image(brightfield, cells, args, save=True)
    
    ### Testing Testing 1 2 3 ###
    # random labels for cells#
    print('Starting label generation...', flush=True)
    codex_key_path = 'data/Section6_ChannelKey.txt'
    kidney_cell_labels, label_colors, tiff_section_names = generate_random_labels(codex_key_path, nuclei)  
    
    print('debug')
    #print(kidney_cell_labels)
    print(label_colors)
    print(tiff_section_names)
    print('end of generate random labels')
    print(len(nuclei))
    
    print('Overlaying nuclei centers...', flush=True)
    #overlay_nuclei_centers(brightfield, nuclei, kidney_cell_labels, label_colors)
    print('end of overlay_nuclei centers')
    num_nuclei = len(nuclei)
    for m in range(0,num_nuclei, 35000):
        end_index = min(m+35000, num_nuclei)
        nuclei_subsample = nuclei[m:end_index]
        make_xml_annotations(tiff_section_names, nuclei_subsample, kidney_cell_labels,m=m)
        print(f'xml file written, m ={m}', flush=True)
        
    #make_xml_annotations(tiff_section_names, nuclei, kidney_cell_labels)
    print('End of XML annotation', flush=True)
#    print('making JSON')
#    make_json_file()
#    print('finished JSON')

    
   
