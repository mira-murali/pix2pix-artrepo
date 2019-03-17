import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import torch


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        '''
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        '''
        self.dirs = []
        self.dirs.append(os.path.join(opt.dataroot, 'A', opt.phase, opt.colorA))
        if opt.input_nc > 3:
            self.dirs.append(os.path.join(opt.dataroot, 'A', opt.phase, opt.bwA))
        self.dirs.append(os.path.join(opt.dataroot, 'B', opt.phase))
        self.dir_paths = []
        for dir in self.dirs:
            self.dir_paths.append(sorted(make_dataset(dir, opt.max_dataset_size)))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.bw_transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        '''
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        '''
        Acolor = Image.open(self.dir_paths[0][index]).convert('RGB')
        B = Image.open(self.dir_paths[-1][index]).convert('RGB')

        '''
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        '''
        # apply the same transform to both A and B
        # transform_params = get_params(self.opt, Acolor.size)
        # Acolor_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # Acolor = Acolor_transform(Acolor)
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        # B = B_transform(B)
        A = self.transforms(Acolor)
        B = self.transforms(B)

        if self.input_nc > 3:
            Abw = Image.open(self.dir_paths[1][index])
            Abw = self.bw_transform(Abw)
            # transform_params_bw = get_params(self.opt, Abw.size)
            # Abw_transform = get_transform(self.opt, transform_params_bw, grayscale=True)
            # Abw = Abw_transform(Abw)
            A = torch.cat((A, Abw), dim=0)

        return {'A': A, 'B': B, 'A_paths': self.dir_paths[:-1], 'B_paths': self.dir_paths[-1]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dir_paths[0])
