import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils import data

class patch_loader(data.Dataset):
    """
        Data loader for the patch-based deconvnet
    """
    def __init__(self, split='train', stride=30 ,patch_size=99, is_transform=True,
                 augmentations=None):
        self.root = 'data/'
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 6 
        self.mean = 0.000941 # average of the training data  
        self.patches = collections.defaultdict(list)
        self.patch_size = patch_size
        self.stride = stride

        if 'test' not in self.split: 
            # Normal train/val mode
            self.seismic = self.pad_volume(np.load(pjoin('data','train','train_seismic.npy')))
            self.labels = self.pad_volume(np.load(pjoin('data','train','train_labels.npy')))
        elif 'test1' in self.split:
            self.seismic = np.load(pjoin('data','test_once','test1_seismic.npy'))
            self.labels = np.load(pjoin('data','test_once','test1_labels.npy'))
        elif 'test2' in self.split:
            self.seismic = np.load(pjoin('data','test_once','test2_seismic.npy'))
            self.labels = np.load(pjoin('data','test_once','test2_labels.npy'))
        else:
            raise ValueError('Unknown split.')

        if 'test' not in self.split:
            # We are in train/val mode. Most likely the test splits are not saved yet, 
            # so don't attempt to load them.  
            for split in ['train', 'val', 'train_val']:
                # reading the file names for 'train', 'val', 'trainval'""
                path = pjoin('data', 'splits', 'patch_' + split + '.txt')
                patch_list = tuple(open(path, 'r'))
                # patch_list = [id_.rstrip() for id_ in patch_list]
                self.patches[split] = patch_list
        elif 'test' in split:
            # We are in test mode. Only read the given split. The other one might not 
            # be available. 
            path = pjoin('data', 'splits', 'patch_' + split + '.txt')
            file_list = tuple(open(path,'r'))
            # patch_list = [id_.rstrip() for id_ in patch_list]
            self.patches[split] = patch_list
        else:
            raise ValueError('Unknown split.')

    def pad_volume(self,volume):
        '''
        Only used for train/val!! Not test.
        '''
        assert 'test' not in self.split, 'There should be no padding for test time!'
        return np.pad(volume,pad_width=self.patch_size,mode='constant', constant_values=255)
        

    def __len__(self):
        return len(self.patches[self.split])

    def __getitem__(self, index):

        patch_name = self.patches[self.split][index]
        direction, idx, xdx, ddx = patch_name.split(sep='_')

        shift = (self.patch_size if 'test' not in self.split else 0)
        idx, xdx, ddx = int(idx)+shift, int(xdx)+shift, int(ddx)+shift

        if direction == 'i':
            im = self.seismic[idx,xdx:xdx+self.patch_size,ddx:ddx+self.patch_size]
            lbl = self.labels[idx,xdx:xdx+self.patch_size,ddx:ddx+self.patch_size]
        elif direction == 'x':    
            im = self.seismic[idx: idx+self.patch_size, xdx, ddx:ddx+self.patch_size]
            lbl = self.labels[idx: idx+self.patch_size, xdx, ddx:ddx+self.patch_size]

        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
            
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl


    def transform(self, img, lbl):
        img -= self.mean

        # to be in the BxCxHxW that PyTorch uses: 
        img, lbl = img.T, lbl.T

        img = np.expand_dims(img,0)
        lbl = np.expand_dims(lbl,0)

        img = torch.from_numpy(img)
        img = img.float()
        lbl = torch.from_numpy(lbl)
        lbl = lbl.long()
                
        return img, lbl

    def get_seismic_labels(self):
        return np.asarray([ [69,117,180], [145,191,219], [224,243,248], [254,224,144], [252,141,89],
                          [215,48,39]])


    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_seismic_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

        
class section_loader(data.Dataset):
    """
        Data loader for the section-based deconvnet
    """
    def __init__(self, split='train', is_transform=True,
                 augmentations=None):
        self.root = 'data/'
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 6 
        self.mean = 0.000941 # average of the training data  
        self.sections = collections.defaultdict(list)

        if 'test' not in self.split: 
            # Normal train/val mode
            self.seismic = np.load(pjoin('data','train','train_seismic.npy'))
            self.labels = np.load(pjoin('data','train','train_labels.npy'))
        elif 'test1' in self.split:
            self.seismic = np.load(pjoin('data','test_once','test1_seismic.npy'))
            self.labels = np.load(pjoin('data','test_once','test1_labels.npy'))
        elif 'test2' in self.split:
            self.seismic = np.load(pjoin('data','test_once','test2_seismic.npy'))
            self.labels = np.load(pjoin('data','test_once','test2_labels.npy'))
        else:
            raise ValueError('Unknown split.')

        if 'test' not in self.split:
            # We are in train/val mode. Most likely the test splits are not saved yet, 
            # so don't attempt to load them.  
            for split in ['train', 'val', 'train_val']:
                # reading the file names for 'train', 'val', 'trainval'""
                path = pjoin('data', 'splits', 'section_' + split + '.txt')
                file_list = tuple(open(path, 'r'))
                file_list = [id_.rstrip() for id_ in file_list]
                self.sections[split] = file_list
        elif 'test' in split:
            # We are in test mode. Only read the given split. The other one might not 
            # be available. 
            path = pjoin('data', 'splits', 'section_' + split + '.txt')
            file_list = tuple(open(path,'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.sections[split] = file_list
        else:
            raise ValueError('Unknown split.')


    def __len__(self):
        return len(self.sections[self.split])

    def __getitem__(self, index):

        section_name = self.sections[self.split][index]
        direction, number = section_name.split(sep='_')

        if direction == 'i':
            im = self.seismic[int(number),:,:]
            lbl = self.labels[int(number),:,:]
        elif direction == 'x':    
            im = self.seismic[:,int(number),:]
            lbl = self.labels[:,int(number),:]
        
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
            
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl


    def transform(self, img, lbl):
        img -= self.mean

        # to be in the BxCxHxW that PyTorch uses: 
        img, lbl = img.T, lbl.T

        img = np.expand_dims(img,0)
        lbl = np.expand_dims(lbl,0)

        img = torch.from_numpy(img)
        img = img.float()
        lbl = torch.from_numpy(lbl)
        lbl = lbl.long()
                
        return img, lbl

    def get_seismic_labels(self):
        return np.asarray([ [69,117,180], [145,191,219], [224,243,248], [254,224,144], [252,141,89],
                          [215,48,39]])


    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_seismic_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb
        
