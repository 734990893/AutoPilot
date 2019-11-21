import numpy as np
from PIL import Image

import os


def process_np_images(target_resolution=(720, 480),
                      path=r"./",
                      target_path=r"./frames"):
    """Rescale np images into target resolution

    Parameters
    ----------
    target_resolution: tuple
        resolution of new images
    path: regexp, optional
        path to the folder where images are located
    target_path: regexp, optional
        path to the folder where new images to store to


    Returns
    -------
    str
        path to the folder where new images are stored
    """
    image_files = [file
                   for file
                   in os.listdir(path)
                   if file.endswith("npy")]

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for image_file in image_files:
        image_set = np.load(image_file)
        image_set_folder = '{}/{}'.format(target_path,
                                          image_file[:-4])
        if not os.path.exists(image_set_folder):
            os.mkdir(image_set_folder)
        for i, img in enumerate(image_set):
            img = Image.fromarray(np.flip(img,
                                          2)).resize(target_resolution)
            img.save('{}/img{}.png'.format(image_set_folder,
                                           str(i).zfill(4)))

    return target_path


if __name__ == "__main__":
    process_np_images()
