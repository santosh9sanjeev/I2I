# import os.path
# from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
# import random
# import util.util as util


# class UnalignedDataset(BaseDataset):
#     """
#     This dataset class can load unaligned/unpaired datasets.

#     It requires two directories to host training images from domain A '/path/to/data/trainA'
#     and from domain B '/path/to/data/trainB' respectively.
#     You can train the model with the dataset flag '--dataroot /path/to/data'.
#     Similarly, you need to prepare two directories:
#     '/path/to/data/testA' and '/path/to/data/testB' during test time.
#     """

#     def __init__(self, opt):
#         """Initialize this dataset class.

#         Parameters:
#             opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseDataset.__init__(self, opt)
#         self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
#         self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

#         if opt.phase == "test" and not os.path.exists(self.dir_A) \
#            and os.path.exists(os.path.join(opt.dataroot, "valA")):
#             self.dir_A = os.path.join(opt.dataroot, "valA")
#             self.dir_B = os.path.join(opt.dataroot, "valB")

#         self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
#         self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
#         self.A_size = len(self.A_paths)  # get the size of dataset A
#         self.B_size = len(self.B_paths)  # get the size of dataset B

#     def __getitem__(self, index):
#         """Return a data point and its metadata information.

#         Parameters:
#             index (int)      -- a random integer for data indexing

#         Returns a dictionary that contains A, B, A_paths and B_paths
#             A (tensor)       -- an image in the input domain
#             B (tensor)       -- its corresponding image in the target domain
#             A_paths (str)    -- image paths
#             B_paths (str)    -- image paths
#         """
#         A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
#         if self.opt.serial_batches:   # make sure index is within then range
#             index_B = index % self.B_size
#         else:   # randomize the index for domain B to avoid fixed pairs.
#             index_B = random.randint(0, self.B_size - 1)
#         B_path = self.B_paths[index_B]
#         A_img = Image.open(A_path).convert('RGB')
#         B_img = Image.open(B_path).convert('RGB')

#         # Apply image transformation
#         # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
#         # do not perform resize-crop data augmentation of CycleGAN.
#         is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
#         modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
#         transform = get_transform(modified_opt)
#         A = transform(A_img)
#         B = transform(B_img)

#         return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

#     def __len__(self):
#         """Return the total number of images in the dataset.

#         As we have two datasets with potentially different number of images,
#         we take a maximum of
#         """
#         return max(self.A_size, self.B_size)


import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.transform import resize
import torchvision
import torch

def load_and_adjust_spatial_nii(nii_file, target_h, target_w):
    nii_image = nib.load(nii_file)
    nii_data = nii_image.get_fdata()
    orig_h, orig_w = nii_data.shape
    pad_h = max(target_h - orig_h, 0)
    pad_w = max(target_w - orig_w, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_data = np.pad(nii_data, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    resized_data = resize(padded_data, (target_h, target_w), mode='constant', preserve_range=True, anti_aliasing=True)

    return resized_data




class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        print(f'Loading data from {self.dir_A} and {self.dir_B}')

        # df = pd.read_csv(opt.dataframe)

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        print(len(self.A_paths), len(self.B_paths))
        # df['Filename'] = '/share/ssddata/CrossMoDA_final_dataset_before_crop_for_GANs/trainB/' + df['Filename']

        # skip_files = df['Filename'].tolist()

        # for file_name in self.B_paths[:]:  # Use a copy of the list to avoid modifying the original list
        #     if file_name in skip_files:
        #         self.B_paths.remove(file_name)   

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        print(self.A_size, self.B_size)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.to_pil = torchvision.transforms.ToPILImage()

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_data = Image.open(A_path)
        B_data = Image.open(B_path)
        # A_img = torch.load(A_path)
        # B_img = torch.load(B_path)

        # A_data = self.to_pil(A_img)
        # B_data = self.to_pil(B_img)

        A = self.transform_A(A_data)
        B = self.transform_B(B_data)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
