import argparse
import logging
import numpy as np
import sys
import cv2
from images import load_images, generate_classified_image
from cells import segment_nuclei_dapi, calculate_radii_from_nuclei, create_cells, extract_nuclei_coordinates, calculate_nuclei_sizes
from visualization import overlay_cell_boundaries, make_xml_annotations, overlay_nuclei_centers, generate_random_labels
from feature_extraction import generate_feature_extractors, calculate_cell_features
from clustering import cluster
from concurrent.futures import ThreadPoolExecutor, wait

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to classify cells from a CODEX file')
    parser.add_argument('--codex', '-c', required=True, help='Path to CODEX tiff file', type=str)
    parser.add_argument('--he', required=True, help='Path to H&E stained tiff file', type=str)
    parser.add_argument('--output', '-o',
                        help='output file (default to ./clustered/classified_stain.tif)',
                        type=str, default='./clustered/clustered_stain.tif')
    parser.add_argument('--dapi', '-d', help='DAPI layer of CODEX file (default to 0)', type=int, default=0)
    parser.add_argument('--n_jobs', '-j', help='Number of CPU cores to run in parallel', type=int, default=4)
    parser.add_argument('--n_workers', '-w', help='Number of workers for multithreading',
                        type=int, default=8)
    parser.add_argument('--verbose', '-v', help='Print debug output', action='store_true')
    parser.add_argument('--scaling_factor', '-s',
                        help='Scaling factor to apply to global threshold. Set higher for larger nuclei.', type=float,
                        default=1.5)

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
    
    _, codex = load_images(args, rotate_brightfield=True)
    

    nuclei_mask_dapi = segment_nuclei_dapi(codex, dapi_index=args.dapi,
                                           scaling_factor=args.scaling_factor, window_size=64, visual_output=False)
    nuclei_dapi = extract_nuclei_coordinates(nuclei_mask_dapi, downsample_factor=4,
                                             num_processes=args.n_jobs, visual_output=False)
                                             
    radii = calculate_radii_from_nuclei(nuclei_dapi, nuclei_mask_dapi, window_size=128)
    cells = create_cells(nuclei_dapi, radii)

    feature_extractors = generate_feature_extractors()
    logger.info('Beginning feature extraction...')
    
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = []
        for cell in cells:
            futures.append(executor.submit(calculate_cell_features,
                                           cell=cell, feature_extractors=feature_extractors, codex=codex))

        wait(futures)
        logger.debug(f'Completed {len(futures)} futures')
        cells = [processed_cell.result() for processed_cell in futures]
    logger.info('Feature extraction complete!')
    cluster(cells)
    channel_last_codex = np.transpose(codex, axes=(1, 2, 0))
    generate_classified_image(channel_last_codex[:, :, 0], cells, args, save=True)

    clustered_nuclei = [cell.nucleus for cell in cells]
    clustered_labels = [cell.label for cell in cells]
    
    '''
    # Brightfield XML annotation with random labels
    logger.info('Starting label generation...')
    codex_key_path = 'data/Section6_ChannelKey.txt'
    kidney_cell_labels, label_colors, tiff_section_names = generate_random_labels(codex_key_path, nuclei_brightfield)  
    

    num_nuclei = len(nuclei_brightfield)
    for m in range(0,num_nuclei, 35000):
        end_index = min(m+35000, num_nuclei)
        nuclei_subsample = nuclei[m:end_index]
        make_xml_annotations(tiff_section_names, nuclei_subsample, kidney_cell_labels,m=m)
        print(f'xml file written, m ={m}', flush=True)
        
    # Whole image XML file
    #make_xml_annotations(tiff_section_names, nuclei, kidney_cell_labels)
    print('End of XML annotation', flush=True)
    '''