import os
import random, math
import torch
import numpy as np
import glob
import jpeg4py as jpeg
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def augment(patch_low_img, patch_high_img, thumb_low_img, thumb_high_img):

	if np.random.randint(2, size=1)[0] == 1:
		patch_low_img = np.flip(patch_low_img, axis=1).copy()
		patch_high_img = np.flip(patch_high_img, axis=1).copy()
		thumb_low_img = np.flip(thumb_low_img, axis=1).copy()
		thumb_high_img = np.flip(thumb_high_img, axis=1).copy()
	if np.random.randint(2, size=1)[0] == 1: 
		patch_low_img = np.flip(patch_low_img, axis=0).copy()
		patch_high_img = np.flip(patch_high_img, axis=0).copy()
		thumb_low_img = np.flip(thumb_low_img, axis=0).copy()
		thumb_high_img = np.flip(thumb_high_img, axis=0).copy()
	if np.random.randint(2, size=1)[0] == 1:
		patch_low_img = np.transpose(patch_low_img, (1, 0, 2)).copy()
		patch_high_img = np.transpose(patch_high_img, (1, 0, 2)).copy()
		thumb_low_img = np.transpose(thumb_low_img, (1, 0, 2)).copy()
		thumb_high_img = np.transpose(thumb_high_img, (1, 0, 2)).copy()

	return patch_low_img, patch_high_img, thumb_low_img, thumb_high_img


class UPE(Dataset):
	def __init__(self, root_dir):
		self.input_dir = os.path.join(root_dir, 'low') 
		self.gt_dir = os.path.join(root_dir, 'high') 
		self.input_thumb_dir = os.path.join(root_dir, 'thumb_low')
		self.gt_thumb_dir = os.path.join(root_dir, 'thumb_high') 

		fns = os.listdir(self.input_dir)
		self.ids = [os.path.basename(fn) for fn in fns]

	def __len__(self):
		l = len(self.ids)
		return l

	def __getitem__(self, idx):
		fns = glob.glob(os.path.join(self.input_dir, self.ids[idx], '*.jpg'))
		filename = os.path.basename(fns[np.random.randint(len(fns))])
		
		H_start, H_end, W_start, W_end, H_bias, W_bias = (filename.split('.')[0]).split('_')

		location = [int(H_end)-int(H_start), int(W_end)-int(W_start), int(H_bias), int(W_bias)]

		low_img = jpeg.JPEG(os.path.join(self.input_dir, self.ids[idx], filename)).decode() / 255.
		high_img = jpeg.JPEG(os.path.join(self.gt_dir, self.ids[idx], filename)).decode() / 255.
		thumb_low_img = jpeg.JPEG(os.path.join(self.input_thumb_dir, self.ids[idx], filename)).decode() / 255.
		thumb_high_img = jpeg.JPEG(os.path.join(self.gt_thumb_dir, self.ids[idx], filename)).decode() / 255.

		low_img, high_img, thumb_low_img, thumb_high_img = augment(low_img, high_img, thumb_low_img, thumb_high_img)

		low_img = hwc_to_chw(low_img)
		high_img = hwc_to_chw(high_img)
		thumb_low_img = hwc_to_chw(thumb_low_img)
		thumb_high_img = hwc_to_chw(thumb_high_img)

		return low_img, high_img, thumb_low_img, thumb_high_img, location


class UPE_INF(Dataset):
	def __init__(self, root_dir, thumb_size=224):
		self.input_dir = os.path.join(root_dir, 'low') 
		self.gt_dir = os.path.join(root_dir, 'high') 

		fns = glob.glob(os.path.join(self.gt_dir, '*.bmp'))
		self.filenames = [os.path.basename(fn) for fn in fns]

		self.thumb_size = thumb_size

		self.random = random

	def __len__(self):
		l = len(self.filenames)
		return l

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		low_img = read_img(os.path.join(self.input_dir, self.filenames[idx]))
		high_img = read_img(os.path.join(self.gt_dir, self.filenames[idx]))
		thumb_low_img = cv2.resize(low_img, (self.thumb_size, self.thumb_size))

		low_img = hwc_to_chw(low_img)
		high_img = hwc_to_chw(high_img)
		thumb_low_img = hwc_to_chw(thumb_low_img)

		return low_img, high_img, thumb_low_img