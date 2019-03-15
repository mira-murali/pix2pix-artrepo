"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
import cv2
from PIL import Image
import os
import glob
import shutil

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def merge_images(dirA, dirB, final_dir):
    """
    dirA contains images of type A
    dirB contains images of type B
    final_dir will store the images pairs {A, B}
    """
    imagesA = glob.glob(os.path.join(dirA, '*'))
    imagesB = glob.glob(os.path.join(dirB, '*'))
    imagesA.sort()
    imagesB.sort()
    for imgA_path, imgB_path in zip(imagesA, imagesB):
        imgA = np.array(Image.open(imgA_path).convert('RGB'), dtype=np.uint8)
        imgB = np.array(Image.open(imgB_path).convert('RGB'), dtype=np.uint8)
        assert imgA.shape == imgB.shape, "Images are not the same size"
        image_pair = np.concatenate((imgA, imgB), axis=1)
        print(imgA.shape, imgB.shape, image_pair.shape)
        img = Image.fromarray(image_pair)
        rev_path = imgA_path[-1::-1]
        last_slash = rev_path.find('/')
        img.save(os.path.join(final_dir, imgA_path[-last_slash:]))

def select_images(images_dir, resized_dir, filename):
    """
    images_dir is the directory all of the ffhq dataset in order
    This function is going to grab the first 10 GB of data and apply blurring
    """
    current_path = os.getcwd()
    os.chdir(images_dir)
    dst_size = 10e9
    images = glob.glob(os.path.join(images_dir, '*'))
    images.sort()
    dir_size = sum(os.path.getsize(f) for f in os.listdir('.') if os.path.isfile(f))
    num_images = int((dst_size/dir_size)*len(images))
    count = 0
    os.chdir(current_path)
    with open(filename, 'w') as f:
        while count < num_images:
            img = cv2.resize(cv2.imread(images[count]), (512, 512))
            median_img = cv2.medianBlur(img, 9)
            for i in range(20):
                median_img = cv2.medianBlur(median_img, 9)
            pyr_img = cv2.pyrMeanShiftFiltering(median_img, 21, 15)
            rev_path = images[count][-1::-1]
            last_slash = rev_path.find('/')
            cv2.imwrite(os.path.join(resized_dir, images[count][-last_slash:]), pyr_img)
            f.write(os.path.join(resized_dir, images[count][-last_slash:])+'\n')
            count += 1

def copy_files(src_dir, dest_dir, path_file='partA.txt'):
    """
    src_dir: contains the files which needs to be copied to dest_dir
    path_file: contains list of file names to be copied from src_dir
    """
    index = -1
    with open(path_file) as filepath:
        img_names = [f.strip('\n') for f in filepath.readlines()]
        for img_name in img_names:
            rev_path = img_name[-1::-1]
            last_slash = rev_path.find('/')
            if last_slash < 0:
                index = 0
            else:
                index = -last_slash
            shutil.copy(os.path.join(src_dir, img_name[index:]), dest_dir)

def train_val_split(src_dir, train_dir, val_dir=None):
    if not val_dir:
        folders = [train_dir]
    else:
        folders = [train_dir, val_dir]
    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)

    all_images = os.listdir(src_dir)
    count = 0
    while count < len(all_images):
        if count + 1 < len(all_images) - 10:
            shutil.copy(os.path.join(src_dir, all_images[count]), train_dir)
        else:
            if val_dir:
                shutil.copy(os.path.join(src_dir, all_images[count]), val_dir)
        count += 1

def create_dir_tree(dataroot = './dataset/'):
    mkdir(os.path.join(dataroot, 'B', 'train'))
    mkdir(os.path.join(dataroot, 'B', 'val'))
    mkdir(os.path.join(dataroot, 'B', 'test'))
    mkdir(os.path.join(dataroot, 'A', 'train', 'blurred_images'))
    mkdir(os.path.join(dataroot, 'A', 'train', 'hed'))
    mkdir(os.path.join(dataroot, 'A', 'val', 'blurred_images'))
    mkdir(os.path.join(dataroot, 'A', 'val', 'hed'))
    mkdir(os.path.join(dataroot, 'A', 'test', 'blurred_images'))
    mkdir(os.path.join(dataroot, 'A', 'test', 'hed'))


