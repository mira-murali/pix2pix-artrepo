import os
from util.util import copy_files
from util.util import select_images

if not os.path.isdir('edges'):
    os.mkdir('edges')

copy_files(src_dir='blurred_images/', dest_dir = 'edges/')