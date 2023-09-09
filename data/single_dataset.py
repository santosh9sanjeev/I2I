# from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image


# class SingleDataset(BaseDataset):
#     """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

#     It can be used for generating CycleGAN results only for one side with the model option '-model test'.
#     """

#     def __init__(self, opt):
#         """Initialize this dataset class.

#         Parameters:
#             opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseDataset.__init__(self, opt)
#         self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
#         input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
#         self.transform = get_transform(opt, grayscale=(input_nc == 1))

#     def __getitem__(self, index):
#         """Return a data point and its metadata information.

#         Parameters:
#             index - - a random integer for data indexing

#         Returns a dictionary that contains A and A_paths
#             A(tensor) - - an image in one domain
#             A_paths(str) - - the path of the image
#         """
#         A_path = self.A_paths[index]
#         A_img = Image.open(A_path).convert('RGB')
#         A = self.transform(A_img)
#         return {'A': A, 'A_paths': A_path}

#     def __len__(self):
#         """Return the total number of images in the dataset."""
#         return len(self.A_paths)
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import nibabel as nib


# from skimage.transform import resize



# def load_and_adjust_spatial_nii(nii_file, target_h, target_w):
#     nii_image = nib.load(nii_file)
#     nii_data = nii_image.get_fdata()
#     orig_h, orig_w = nii_data.shape
#     pad_h = max(target_h - orig_h, 0)
#     pad_w = max(target_w - orig_w, 0)

#     pad_top = pad_h // 2
#     pad_bottom = pad_h - pad_top
#     pad_left = pad_w // 2
#     pad_right = pad_w - pad_left

#     padded_data = np.pad(nii_data, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
#     resized_data = resize(padded_data, (target_h, target_w), mode='constant', preserve_range=True, anti_aliasing=True)

#     return resized_data



class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform_A = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]

        A_img = nib.load(A_path)
        A_data = A_img.get_fdata()
        # print('herllllllllllllllllloooooooooooo', A_data.shape)
        # A_data = load_and_adjust_spatial_nii(A_path, 352, 352)
        A_data = (A_data - A_data.min())/(A_data.max()-A_data.min()) * 255
        
        A_data = Image.fromarray(np.uint8(A_data))#.convert('RGB')
        A = self.transform_A(A_data)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
