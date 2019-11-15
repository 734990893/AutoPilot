from PIL import Image

import os
import sys

def preprocess_images(extension: str, 
                      path=r"./", 
                      target_path=r"./preprocessed"):
    """Rescale images into 180px x 180px
    
    Parameters
    ----------
    extension: str
        target image extension
    path: regexp, optional
        path to the folder where images are located
    target_path: regexp, optional
        path to the folder where new images to store to
    

    Returns
    -------
    str
        path to the folder where new images are stored
    """
    ext = '.{}'.format(extension)
    image_files = [file 
                   for file 
                   in os.listdir(path) 
                   if file.endswith(ext)]
    
    if not os.path.exists(target_path):
        os.mkdir(target_path)
        
    target_resolution = (180, 180)
    for image_file in image_files:
        with Image.open(image_file) as img:
            store_loc = '{}/{}'.format(target_path, 
                                       image_file)
            img.resize(target_resolution).save(store_loc)
        
    return target_path

if __name__ == "__main__":
    image_extension = sys.argv[1]
    preprocess_images(image_extension)
