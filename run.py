import os
from util.util import copy_files
from util.util import select_images

if not os.path.isdir('test_dataset/hed'):
    os.makedirs('test_dataset/hed')

copy_files(src_dir='dataset/hed', dest_dir = 'test_dataset/hed', path_file='facetest.txt')

'''
select_images(images_dir='test_dataset/orig_images', resized_dir='test_dataset/blurred_images', filename='facetest.txt')
'''