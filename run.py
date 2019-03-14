import os
from util.util import select_images

if not os.path.isdir('blurred_images'):
    os.mkdir('blurred_images')

select_images(images_dir='/media/nanda/Data/Research Data/Mira/ffhq-dataset/images1024x1024', resized_dir = 'blurred_images')