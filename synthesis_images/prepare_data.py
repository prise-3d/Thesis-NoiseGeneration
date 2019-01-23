from ipfml import processing
from PIL import Image

import shutil
import os

images_folder = "images"
dest_folder = "generated_blocks"

if os.path.exists(dest_folder):
    # first remove folder if necessary
    os.rmdir(os.path.abspath(os.path.join(os.getcwd(), dest_folder)))

# create new images
images = os.listdir(images_folder)

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

for img_path in images:

    img = Image.open(os.path.join(images_folder, img_path))

    blocks = processing.divide_in_blocks(img, (80, 80), pil=True)

    for id, pil_block in enumerate(blocks):
        img_name = img_path.split('/')[-1]
        split_name = img_name.split('.')
        block_name = split_name[0] + "_" + str(id) + "." + split_name[1]

        dest_img_block = os.path.join(dest_folder, block_name)

        pil_block.save(dest_img_block)


