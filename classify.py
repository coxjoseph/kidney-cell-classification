import argparse
import logging
import numpy as np
from images import load_images, generate_classified_image
from cells import segment_nuclei_brightfield, segment_nuclei_dapi, calculate_radii_from_nuclei, create_cells, extract_nuclei_coordinates, create_nuclei, get_nucleus_mask, Nucleus
from visualization import overlay_cell_boundaries
from skimage import io, transform
from feature_extraction import generate_feature_extractors
from clustering import cluster
from feature_extraction import get_nuclei_size
import cv2
import wsi_annotations_kit.wsi_annotations_kit as wak
from shapely.geometry import Point
import random as rd
import seaborn as sns

logger = logging.getLogger()

###########################################################################################################################
def overlay_nuclei_centers(image_data, nuclei: list[Nucleus], labels, label_colors):
    num_nuclei = len(nuclei)
    dots_image = np.zeros_like(image_data)
    
    for i in range(num_nuclei):
        x = nuclei[i].center[1]
        y= nuclei[i].center[0]
        color = label_colors[labels[i]] if labels[i] < len(label_colors) else (0, 0, 0)  # Default color is black for unknown labels
        
        cv2.circle(dots_image, (int(x), int(y)), 10, color, -1)  # Increase dot size

        # Overlay the dots image onto the original glomeruli image with increased opacity
        result_image = cv2.addWeighted(image_data, 0.35, dots_image, 1, 0)
        
    cv2.imshow('Nuclei Centers Overlay', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    return result_image


def generate_random_labels(codex_path, nuclei):
    
    with open(codex_path, 'r') as file:
        codex_names = file.readlines()
        codex_names = [name.strip().split(maxsplit=1)[-1] for name in codex_names]
        
        color_palette = sns.color_palette("colorblind", len(codex_names)) # Generate colorblind-friendly palette with seaborn
        rgb_values = [(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in color_palette] # Convert seaborn color palette to RGB values
        
        codex_channels = np.linspace(0, len(nuclei)-1,dtype=int)
        labels = []
        
        for index in nuclei:
            cell_label = rd.randint(0, len(codex_names)-1)
            labels.append(cell_label)
    
    return labels, rgb_values, codex_names


def make_xml_annotations(cell_names, nuclei_centers, labels, filename='Brightfield_XML_Annot.xml'):
    annotations = wak.Annotation()
    annotations.add_names(cell_names)
    radius = 5

    for ii in range(len(nuclei_centers)):
        y = nuclei_centers[ii][1]
        x = nuclei_centers[ii][0]
        point = Point(y, x)
        circle = point.buffer(5).simplify(tolerance=0.05, preserve_topology=False)
        annotations.add_shape(poly=circle, box_crs=[0, 0], structure=cell_names[labels[ii]-1], name=f"Cell {ii}") #update crs if they're relative to smaller mask to be top left corner
    print(annotations)
    annotations.xml_save(filename)
    return None

    

###############################################################################################################################



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
    
    nuclei_mask = segment_nuclei_dapi(codex, DAPI_index=args.dapi)
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
    
    codex_key_path = 'data/Section6_ChannelKey.txt'
    
    kidney_cell_labels, label_colors, tiff_section_names = generate_random_labels(codex_key_path, nuclei)  
    
    print('debug')
    print(kidney_cell_labels)
    print(label_colors)
    print(tiff_section_names)
    print('end of generate random labels')
    print(len(nuclei))
    
    overlay_nuclei_centers(brightfield, nuclei, kidney_cell_labels, label_colors)
    print('end of overlay_nuclei centers')
    make_xml_annotations(tiff_section_names, nuclei, kidney_cell_labels)

    
   
