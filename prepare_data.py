from util.util import create_dir_tree
from util.util import train_val_split
import sys
import os

dataroot = sys.argv[1]
data_dir = sys.argv[2]

if not os.path.isdir(dataroot):
    os.mkdir(dataroot)
# The following command creates the directory structure specified in README when provided with the parent directory
create_dir_tree(dataroot)

#Creates train-val split
train_val_split(src_dir = os.path.join(data_dir, 'blurred_images'), train_dir = os.path.join(dataroot, 'A', 'train', 'blurred_images'),
                val_dir = os.path.join(dataroot, 'A', 'val', 'blurred_images'))
train_val_split(src_dir = os.path.join(data_dir, 'hed'), train_dir = os.path.join(dataroot, 'A', 'train', 'hed'),
                val_dir = os.path.join(dataroot, 'A', 'val', 'hed'))
train_val_split(src_dir = os.path.join(data_dir, 'orig_images'), train_dir = os.path.join(dataroot, 'B', 'train'),
                val_dir = os.path.join(dataroot, 'B', 'val'))
